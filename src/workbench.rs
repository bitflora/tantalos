//! A browser app for triaging the downloaded Workshop levels.
//!
//! `glados-workbench` reads the Workshop corpus (`out/workshop/levels.jsonl`), the ws5
//! train/test splits, the Workshop ratings and item metadata, and serves a local app that
//! shows every field we have on a level, its solver verdict, and its geometry. Notes, a
//! personal star rating, a "staged for playtesting" flag and an "exclude from the test data"
//! flag are saved in `train/workshop-annotations.jsonl` (one JSON object per annotated level).
//!
//! The app is deliberately read-mostly: the level files are never rewritten. The exclusion
//! flag is read by `tools/workshop/ws3.py` (so a rebuilt split moves it out of test) and by
//! `train/plans.py` (so `test-ws` leaves it out).
//!
//! `--generated` points the same app at the level net's own levels instead: a run's eval
//! samples (`runs/RUN/eval/{samples,rated-samples}.jsonl`, and the same under `away/`; the
//! newest level run by default) plus any `--levels FILE`. A level's id is a hash of its tokens.
//! Annotations go to `train/generated-annotations.jsonl` and carry a human verdict (solvable,
//! unsolvable, broken); an unsolvable or broken one is also written to
//! `data/ingame/human-unsolvable.jsonl`, which `train/run.py` trains away from.
//!
//! Endpoints:
//! - `GET /`                    the app
//! - `GET /api/config`          which mode the app is in (workshop or generated), and its files
//! - `GET /api/levels`          every level, merged across the corpus and the ws5 splits
//! - `GET /api/ratings`         Workshop `{id: {stars, votes}}`
//! - `GET /api/items`           Workshop item metadata (`items.jsonl`) by id
//! - `GET /api/fetched`         download status (`fetched.jsonl`) by id
//! - `GET /api/annotations`     our annotations, one object per level
//! - `GET /api/level?id=ID`     one level's tokens and decoded geometry
//! - `POST /api/annotate`       update one annotation (form fields: id, staged, rating,
//!                              complexity, human_solved, human_verdict, notes, excluded)
//! - `POST /api/stage`          compile one level and copy it into the Puzzle Maker folder
//!                              (form fields: id, force); marks it staged
//! - `GET /api/rescan`          re-read the files, then return the levels
//! - `GET /api/export`          the annotations as JSONL, for a backup

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::dataset::json_str;
use crate::level::Level;
use crate::tokens;
use crate::view::geometry_json;

const APP: &str = include_str!("workbench.html");

const USAGE: &str = "usage:
  glados-workbench [--all FILE] [--train FILE] [--test FILE] [--ratings FILE]
                   [--items FILE] [--fetched FILE] [--annotations FILE]
                   [--port N] [--no-open] [--list]
                         a browser app for the downloaded Workshop levels: every field we
                         have, the solver's verdict, the Workshop rating and the level drawn
                         in 2.5D. Rate, note, stage into the Puzzle Maker, or exclude from
                         test data; annotations are saved to
                         train/workshop-annotations.jsonl.
                         (defaults: out/workshop/levels.jsonl, data/ws5/{train,test}.jsonl,
                         out/workshop/{ratings,items,fetched}.jsonl)
  glados-workbench --generated [--run NAME] [--levels FILE]... [--annotations FILE]
                   [--human-unsolvable FILE] [--port N] [--no-open] [--list]
                         the same app for the level net's own levels: the eval samples of a
                         run (default: the newest level run in runs/) and any --levels files
                         (JSONL with a tokens field). Rate, note, stage, and give a human
                         verdict; annotations are saved to train/generated-annotations.jsonl,
                         and unsolvable/broken levels are added to
                         data/ingame/human-unsolvable.jsonl (trained away from).";

const DEFAULT_ALL: &str = "out/workshop/levels.jsonl";
const DEFAULT_TRAIN: &str = "data/ws5/train.jsonl";
const DEFAULT_TEST: &str = "data/ws5/test.jsonl";
const DEFAULT_RATINGS: &str = "out/workshop/ratings.jsonl";
const DEFAULT_ITEMS: &str = "out/workshop/items.jsonl";
const DEFAULT_FETCHED: &str = "out/workshop/fetched.jsonl";
const DEFAULT_ANNOTATIONS: &str = "train/workshop-annotations.jsonl";
const DEFAULT_GEN_ANNOTATIONS: &str = "train/generated-annotations.jsonl";
const DEFAULT_GEN_REGISTRY: &str = "train/generated-staged.jsonl";
const DEFAULT_HUMAN_UNSOLVABLE: &str = "data/ingame/human-unsolvable.jsonl";
const DEFAULT_RUNS: &str = "runs";
/// Where `human-unsolvable.jsonl` rows written by this app say they came from; only these
/// rows are ever removed again.
const HUMAN_SOURCE: &str = "glados-workbench";

/// The files the app reads.
#[derive(Clone)]
pub struct Paths {
    pub all: PathBuf,
    pub train: PathBuf,
    pub test: PathBuf,
    pub ratings: PathBuf,
    pub items: PathBuf,
    pub fetched: PathBuf,
    pub annotations: PathBuf,
    /// Set in `--generated` mode: read the level net's levels instead of the Workshop's.
    pub generated: Option<Generated>,
}

/// The files of `--generated` mode.
#[derive(Clone)]
pub struct Generated {
    /// The run whose eval samples are read (its name under `runs/`), if any.
    pub run: Option<String>,
    pub run_dir: Option<PathBuf>,
    /// More JSONL files with a `tokens` field.
    pub extra: Vec<PathBuf>,
    /// Where unsolvable/broken verdicts are added for the away training.
    pub human_unsolvable: PathBuf,
}

impl Default for Paths {
    fn default() -> Self {
        Paths {
            all: DEFAULT_ALL.into(),
            train: DEFAULT_TRAIN.into(),
            test: DEFAULT_TEST.into(),
            ratings: DEFAULT_RATINGS.into(),
            items: DEFAULT_ITEMS.into(),
            fetched: DEFAULT_FETCHED.into(),
            annotations: DEFAULT_ANNOTATIONS.into(),
            generated: None,
        }
    }
}

/// One level, merged from the corpus and the ws5 split. `fields` holds JSON values verbatim,
/// so fields we don't know about still reach the page.
#[derive(Clone)]
struct Record {
    id: String,
    fields: BTreeMap<String, String>,
    tokens: String,
    split: Option<&'static str>,
    sources: Vec<String>,
    /// Generated levels: "RUN SOURCE line N" of the first file that had it.
    origin: String,
}

/// The user's own triage record for a level.
#[derive(Clone, Default, PartialEq)]
pub struct Annotation {
    pub staged: bool,
    pub rating: i64,
    /// How complex the level is, 0 (unset) to 5; set by the user.
    pub complexity: i64,
    /// The user solved it by hand in Portal 2.
    pub human_solved: bool,
    pub notes: String,
    pub excluded: bool,
    /// Generated levels: "", "solvable", "unsolvable" or "broken", from a human play-test.
    pub human_verdict: String,
    /// Generated levels: the level itself and where it came from, so the store stands alone
    /// after the run is deleted. Not part of `is_empty`.
    pub tokens: String,
    pub origin: String,
    pub updated: i64,
}

const HUMAN_VERDICTS: [&str; 3] = ["solvable", "unsolvable", "broken"];

impl Annotation {
    fn is_empty(&self) -> bool {
        !self.staged && self.rating == 0 && self.complexity == 0 && !self.human_solved && self.notes.is_empty() && !self.excluded
            && self.human_verdict.is_empty()
    }

    /// A verdict that sends the level to `human-unsolvable.jsonl`.
    fn is_bad(&self) -> bool {
        self.human_verdict == "unsolvable" || self.human_verdict == "broken"
    }
}

impl Paths {
    /// The annotation store's id field: Workshop ids, or our token hashes.
    fn id_key(&self) -> &'static str {
        if self.generated.is_some() {
            "id"
        } else {
            "workshop_id"
        }
    }
}

struct State {
    records: Vec<Record>,
    index: HashMap<String, usize>,
    levels_json: String,
    ratings_json: String,
    items_json: String,
    fetched_json: String,
    annotations: HashMap<String, Annotation>,
    paths: Paths,
    /// True while a `POST /api/stage` request is running, so a second click cannot make a
    /// duplicate copy.
    staging: bool,
}

/// `glados-workbench`: parse the flags, then serve (or just print a summary).
pub fn run(args: &[String]) -> Result<(), String> {
    let mut paths = Paths::default();
    let mut port = 0u16;
    let mut open = true;
    let mut list = false;
    let mut generated = false;
    let mut run_name: Option<String> = None;
    let mut levels: Vec<PathBuf> = Vec::new();
    let mut annotations: Option<PathBuf> = None;
    let mut human_unsolvable = PathBuf::from(DEFAULT_HUMAN_UNSOLVABLE);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = |it: &mut std::slice::Iter<String>| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--all" => paths.all = PathBuf::from(value(&mut it)?),
            // The corpus in Workshop mode (an old alias of --all); more files with --generated.
            "--levels" => levels.push(PathBuf::from(value(&mut it)?)),
            "--generated" => generated = true,
            "--run" => run_name = Some(value(&mut it)?),
            "--human-unsolvable" => human_unsolvable = PathBuf::from(value(&mut it)?),
            "--train" => paths.train = PathBuf::from(value(&mut it)?),
            "--test" => paths.test = PathBuf::from(value(&mut it)?),
            "--ratings" => paths.ratings = PathBuf::from(value(&mut it)?),
            "--items" => paths.items = PathBuf::from(value(&mut it)?),
            "--fetched" => paths.fetched = PathBuf::from(value(&mut it)?),
            "--annotations" => annotations = Some(PathBuf::from(value(&mut it)?)),
            "--port" => {
                let v = value(&mut it)?;
                port = v
                    .parse()
                    .map_err(|_| format!("--port: not a number: {v}"))?;
            }
            "--no-open" => open = false,
            "--list" => list = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown flag {flag}\n{USAGE}")),
        }
    }
    if generated {
        let run_dir = match &run_name {
            Some(name) => {
                let dir = Path::new(DEFAULT_RUNS).join(name);
                if !dir.is_dir() {
                    return Err(format!("no run {name} (no folder {})", dir.display()));
                }
                Some(dir)
            }
            None if levels.is_empty() => Some(newest_level_run(Path::new(DEFAULT_RUNS)).ok_or(
                "no level run with eval samples in runs/ (pass --run NAME or --levels FILE)",
            )?),
            None => None,
        };
        let run = run_dir.as_ref().map(|d| {
            d.strip_prefix(DEFAULT_RUNS)
                .unwrap_or(d)
                .to_string_lossy()
                .replace('\\', "/")
        });
        paths.annotations = annotations.unwrap_or_else(|| DEFAULT_GEN_ANNOTATIONS.into());
        paths.generated = Some(Generated {
            run,
            run_dir,
            extra: levels,
            human_unsolvable,
        });
    } else {
        if run_name.is_some() {
            return Err("--run needs --generated".into());
        }
        if let Some(last) = levels.pop() {
            paths.all = last;
        }
        if let Some(a) = annotations {
            paths.annotations = a;
        }
    }
    let state = State::load(&paths)?;
    if list {
        print_summary(&state);
        return Ok(());
    }
    serve(state, port, open)
}

fn serve(state: State, port: u16, open: bool) -> Result<(), String> {
    print_summary(&state);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("can't listen on 127.0.0.1:{port}: {e}"))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?.port();
    let url = format!("http://127.0.0.1:{actual}/");
    println!("  {url}   (Ctrl-C to stop)");
    if open {
        open_browser(&url);
    }
    let state = Arc::new(Mutex::new(state));
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            let _ = handle(stream, &state);
        });
    }
    Ok(())
}

fn print_summary(state: &State) {
    let mut verdicts: BTreeMap<String, usize> = BTreeMap::new();
    let mut annotated = 0;
    let mut staged = 0;
    let mut excluded = 0;
    for r in &state.records {
        let v = r
            .fields
            .get("solver_verdict")
            .or_else(|| r.fields.get("verdict"))
            .map(|v| json_unescape(v.trim_matches('"')))
            .unwrap_or_else(|| "?".into());
        *verdicts.entry(v).or_default() += 1;
        if let Some(a) = state.annotations.get(&r.id) {
            annotated += 1;
            staged += a.staged as usize;
            excluded += a.excluded as usize;
        }
    }
    let by_verdict = verdicts
        .iter()
        .map(|(k, n)| format!("{k} {n}"))
        .collect::<Vec<_>>()
        .join(", ");
    match &state.paths.generated {
        Some(g) => println!(
            "glados-workbench: {} generated levels from {} ({by_verdict})",
            state.records.len(),
            generated_files(g)
                .iter()
                .filter(|(p, _)| p.is_file())
                .map(|(p, _)| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => println!(
            "glados-workbench: {} levels ({by_verdict})",
            state.records.len()
        ),
    }
    let mut verdicts: BTreeMap<&str, usize> = BTreeMap::new();
    for a in state.annotations.values() {
        if !a.human_verdict.is_empty() {
            *verdicts.entry(&a.human_verdict).or_default() += 1;
        }
    }
    let human = verdicts
        .iter()
        .map(|(k, n)| format!(", {n} {k}"))
        .collect::<String>();
    println!(
        "  {} annotations ({staged} staged, {excluded} excluded{human}) -> {}",
        annotated,
        state.paths.annotations.display()
    );
}

fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

impl State {
    fn load(paths: &Paths) -> Result<State, String> {
        let records = match &paths.generated {
            Some(g) => load_generated_levels(g)?,
            None => load_levels(paths)?,
        };
        if records.is_empty() {
            return Err(match &paths.generated {
                Some(g) => format!(
                    "no levels with tokens found in {}",
                    generated_files(g)
                        .iter()
                        .map(|(p, _)| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                None => format!(
                    "no levels found: expected {} (or pass --all/--train/--test)",
                    paths.all.display()
                ),
            });
        }
        // The Workshop metadata means nothing for our own levels.
        let workshop = paths.generated.is_none();
        let meta = |load: &dyn Fn() -> String| if workshop { load() } else { "{}".to_string() };
        let mut index = HashMap::new();
        for (i, r) in records.iter().enumerate() {
            index.insert(r.id.clone(), i);
        }
        let levels_json = format!(
            "{{\"levels\":[{}]}}",
            records
                .iter()
                .map(record_json)
                .collect::<Vec<_>>()
                .join(",")
        );
        let annotations = load_annotations(&paths.annotations)?;
        Ok(State {
            records,
            index,
            levels_json,
            ratings_json: meta(&|| load_ratings(&paths.ratings)),
            items_json: meta(&|| load_map(&paths.items, "id")),
            fetched_json: meta(&|| load_map(&paths.fetched, "id")),
            annotations,
            paths: paths.clone(),
            staging: false,
        })
    }

    fn rescan(&mut self) -> Result<(), String> {
        *self = State::load(&self.paths)?;
        Ok(())
    }
}

fn load_levels(paths: &Paths) -> Result<Vec<Record>, String> {
    let mut merged: BTreeMap<String, Record> = BTreeMap::new();
    add_level_file(&mut merged, &paths.all, "workshop", None)?;
    add_level_file(&mut merged, &paths.train, "ws5-train", Some("train"))?;
    add_level_file(&mut merged, &paths.test, "ws5-test", Some("test"))?;
    Ok(merged.into_values().collect())
}

fn add_level_file(
    merged: &mut BTreeMap<String, Record>,
    path: &Path,
    source: &'static str,
    split: Option<&'static str>,
) -> Result<(), String> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(()), // optional input; the caller reports if nothing loaded at all
    };
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = object_fields(&line);
        let Some(raw_id) = field(&fields, "workshop_id") else {
            continue;
        };
        let id = json_unescape(raw_id.trim_matches('"'));
        if id.is_empty() {
            continue;
        }
        let tokens = field(&fields, "tokens")
            .map(|v| json_unescape(v.trim_matches('"')))
            .unwrap_or_default();
        let rec = merged.entry(id.clone()).or_insert_with(|| Record {
            id: id.clone(),
            fields: BTreeMap::new(),
            tokens: String::new(),
            split: None,
            sources: Vec::new(),
            origin: String::new(),
        });
        for (k, v) in &fields {
            if *k != "tokens" {
                rec.fields.insert((*k).to_string(), (*v).to_string());
            }
        }
        if !tokens.is_empty() {
            rec.tokens = tokens;
        }
        if let Some(s) = split {
            rec.split = Some(s);
        }
        if !rec.sources.iter().any(|s| s == source) {
            rec.sources.push(source.to_string());
        }
    }
    Ok(())
}

/// The newest run started by `train/run.py` (by its status.json `started`) that is a level
/// run with eval samples, like `run.py newest()` but skipping Goblin runs.
fn newest_level_run(runs: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<(String, PathBuf)>) {
        let status = dir.join("status.json");
        if let Ok(text) = std::fs::read_to_string(&status) {
            let fields = object_fields(&text);
            let task = field(&fields, "task").map(|v| json_unescape(v.trim().trim_matches('"')));
            let level_run = matches!(task.as_deref(), None | Some("levels"));
            let has_samples = ["eval/samples.jsonl", "eval/rated-samples.jsonl"]
                .iter()
                .any(|f| dir.join(f).is_file());
            if let (true, true, Some(started)) = (level_run, has_samples, field(&fields, "started")) {
                out.push((json_unescape(started.trim().trim_matches('"')), dir.to_path_buf()));
            }
        }
        if depth == 0 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                if e.file_type().map_or(false, |t| t.is_dir()) {
                    walk(&e.path(), depth - 1, out);
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(runs, 2, &mut found);
    found.into_iter().max().map(|(_, dir)| dir)
}

/// Every file `--generated` reads, with the source name it is shown under.
fn generated_files(g: &Generated) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    if let Some(dir) = &g.run_dir {
        for (sub, prefix) in [(dir.join("eval"), ""), (dir.join("away").join("eval"), "away-")] {
            out.push((sub.join("samples.jsonl"), format!("{prefix}eval")));
            out.push((sub.join("rated-samples.jsonl"), format!("{prefix}rated")));
        }
    }
    for p in &g.extra {
        let name = p
            .file_stem()
            .map_or_else(|| p.display().to_string(), |s| s.to_string_lossy().into_owned());
        out.push((p.clone(), name));
    }
    out
}

/// A level's token string with its spacing normalised, for hashing and comparing.
fn normalize_tokens(tokens: &str) -> String {
    tokens.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A generated level's id: 64-bit FNV-1a of its normalised tokens, as 16 hex digits. The same
/// level gets the same id in every file and run, so its annotation follows it.
fn token_id(tokens: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in normalize_tokens(tokens).bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The level net's levels: a run's eval and rated samples (and its away run's), plus any
/// `--levels` files. Levels without tokens are skipped; the same level in several files is one
/// record listing every source. The Portal 2 editor check (`eval/ingame/results.jsonl`) is
/// joined by tokens into a `game` field.
fn load_generated_levels(g: &Generated) -> Result<Vec<Record>, String> {
    let run = g.run.clone().unwrap_or_default();
    let mut merged: BTreeMap<String, Record> = BTreeMap::new();
    for (path, source) in generated_files(g) {
        let file = match std::fs::File::open(&path) {
            Ok(file) => file,
            Err(_) => continue,
        };
        // `--levels` files are named after themselves, not the run.
        let from_run = !g.extra.contains(&path);
        for (n, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
            let fields = object_fields(&line);
            let tokens = field(&fields, "tokens")
                .map(|v| normalize_tokens(&json_unescape(v.trim_matches('"'))))
                .unwrap_or_default();
            if tokens.is_empty() || tokens == "null" {
                continue;
            }
            let id = token_id(&tokens);
            let line_no = field(&fields, "line")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(n + 1);
            let label = if from_run { format!("{run} {source}") } else { source.clone() };
            let rec = merged.entry(id.clone()).or_insert_with(|| {
                let setting = field(&fields, "setting")
                    .map(|v| json_unescape(v.trim_matches('"')))
                    .filter(|s| !s.is_empty() && s != "null");
                let title = match setting {
                    Some(s) => format!("{label} #{line_no} · {s}"),
                    None => format!("{label} #{line_no}"),
                };
                let mut base = BTreeMap::new();
                base.insert("title".to_string(), json_str(&title));
                base.insert("run".to_string(), json_str(if from_run { &run } else { "" }));
                base.insert("source".to_string(), json_str(&source));
                base.insert("line".to_string(), line_no.to_string());
                Record {
                    id: id.clone(),
                    fields: base,
                    tokens: tokens.clone(),
                    split: None,
                    sources: Vec::new(),
                    origin: format!("{label} line {line_no}"),
                }
            });
            for (k, v) in &fields {
                // `sample` is the raw text the tokens were parsed from: big and redundant.
                if matches!(*k, "tokens" | "sample" | "line" | "title" | "source" | "run") {
                    continue;
                }
                rec.fields.entry((*k).to_string()).or_insert_with(|| v.trim().to_string());
            }
            // The eval file says `verdict`, the rated file `outcome`; the page reads
            // `solver_verdict`.
            if !rec.fields.contains_key("solver_verdict") {
                if let Some(v) = field(&fields, "verdict").or_else(|| field(&fields, "outcome")) {
                    rec.fields.insert("solver_verdict".into(), v.trim().to_string());
                }
            }
            if !rec.sources.iter().any(|s| *s == source) {
                rec.sources.push(source.clone());
            }
        }
    }
    if let Some(dir) = &g.run_dir {
        for sub in ["eval/ingame/results.jsonl", "away/eval/ingame/results.jsonl"] {
            let Ok(file) = std::fs::File::open(dir.join(sub)) else {
                continue;
            };
            for line in BufReader::new(file).lines().flatten() {
                let fields = object_fields(&line);
                let (Some(tokens), Some(game)) = (field(&fields, "tokens"), field(&fields, "game")) else {
                    continue;
                };
                let id = token_id(&json_unescape(tokens.trim_matches('"')));
                if let Some(rec) = merged.get_mut(&id) {
                    rec.fields.insert("game".into(), game.trim().to_string());
                }
            }
        }
    }
    Ok(merged.into_values().collect())
}

/// The compact record the page gets: everything in the file except the tokens, plus which
/// sources carried it and the ws5 split it is in.
fn record_json(r: &Record) -> String {
    let mut parts: Vec<String> = r
        .fields
        .iter()
        .map(|(k, v)| format!("{}:{v}", json_str(k)))
        .collect();
    parts.push(format!(
        "\"sources\":[{}]",
        r.sources
            .iter()
            .map(|s| json_str(s))
            .collect::<Vec<_>>()
            .join(",")
    ));
    parts.push(format!(
        "\"split\":{}",
        r.split.map_or("null".to_string(), json_str)
    ));
    parts.push(format!("\"in_ws5\":{}", r.split.is_some()));
    parts.push(format!("\"has_tokens\":{}", !r.tokens.is_empty()));
    parts.push(format!("\"id\":{}", json_str(&r.id)));
    format!("{{{}}}", parts.join(","))
}

fn load_ratings(path: &Path) -> String {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    if let Ok(file) = std::fs::File::open(path) {
        for line in BufReader::new(file).lines().flatten() {
            let fields = object_fields(&line);
            let Some(raw) = field(&fields, "ratings") else {
                continue;
            };
            for (id, value) in object_fields(raw) {
                if !id.is_empty() {
                    map.insert(json_unescape(id.trim_matches('"')), value.to_string());
                }
            }
        }
    }
    map_json(map)
}

/// A JSON object built from the `id` field of each JSONL line (used for `items.jsonl` and
/// `fetched.jsonl`, whose ids are plain strings).
fn load_map(path: &Path, id_key: &str) -> String {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    if let Ok(file) = std::fs::File::open(path) {
        for line in BufReader::new(file).lines().flatten() {
            let fields = object_fields(&line);
            let Some(raw_id) = field(&fields, id_key) else {
                continue;
            };
            let id = json_unescape(raw_id.trim_matches('"'));
            if id.is_empty() {
                continue;
            }
            let body = fields
                .iter()
                .map(|(k, v)| format!("{}:{v}", json_str(k)))
                .collect::<Vec<_>>()
                .join(",");
            map.insert(id, format!("{{{body}}}"));
        }
    }
    map_json(map)
}

fn map_json(map: BTreeMap<String, String>) -> String {
    format!(
        "{{{}}}",
        map.into_iter()
            .map(|(k, v)| format!("{}:{v}", json_str(&k)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn load_annotations(path: &Path) -> Result<HashMap<String, Annotation>, String> {
    let mut out = HashMap::new();
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(out),
    };
    let text = |fields: &[(&str, &str)], key: &str| {
        field(fields, key)
            .map(|v| json_unescape(v.trim_matches('"')))
            .unwrap_or_default()
    };
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = object_fields(&line);
        // The Workshop store is keyed by `workshop_id`, the generated-level store by `id`.
        let Some(raw_id) = field(&fields, "workshop_id").or_else(|| field(&fields, "id")) else {
            return Err(format!("{}:{}: no workshop_id or id", path.display(), n + 1));
        };
        let id = json_unescape(raw_id.trim_matches('"'));
        out.insert(
            id,
            Annotation {
                staged: field(&fields, "staged").map_or(false, |v| v == "true"),
                rating: field(&fields, "rating")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                complexity: field(&fields, "complexity")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
                human_solved: field(&fields, "human_solved").map_or(false, |v| v == "true"),
                notes: text(&fields, "notes"),
                excluded: field(&fields, "excluded").map_or(false, |v| v == "true"),
                human_verdict: text(&fields, "human_verdict"),
                tokens: text(&fields, "tokens"),
                origin: text(&fields, "origin"),
                updated: field(&fields, "updated")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            },
        );
    }
    Ok(out)
}

fn save_annotations(
    path: &Path,
    key: &str,
    map: &HashMap<String, Annotation>,
) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    let mut ids: Vec<&String> = map.keys().collect();
    ids.sort();
    let mut text = String::new();
    for id in ids {
        let a = &map[id];
        if a.is_empty() {
            continue;
        }
        text.push_str(&annotation_json(key, id, a, true));
        text.push('\n');
    }
    write_atomic(path, &text)
}

fn write_atomic(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("jsonl.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// One annotation as JSON. The generated-level fields are only written when set, so the
/// Workshop store keeps its old shape; `full` adds the level's tokens (the file has them, the
/// page doesn't need them).
fn annotation_json(key: &str, id: &str, a: &Annotation, full: bool) -> String {
    let mut extra = String::new();
    if !a.human_verdict.is_empty() {
        extra.push_str(&format!(",\"human_verdict\":{}", json_str(&a.human_verdict)));
    }
    if !a.origin.is_empty() {
        extra.push_str(&format!(",\"origin\":{}", json_str(&a.origin)));
    }
    if full && !a.tokens.is_empty() {
        extra.push_str(&format!(",\"tokens\":{}", json_str(&a.tokens)));
    }
    format!(
        "{{{}:{},\"staged\":{},\"rating\":{},\"complexity\":{},\"human_solved\":{},\"notes\":{},\"excluded\":{}{extra},\"updated\":{}}}",
        json_str(key),
        json_str(id),
        a.staged,
        a.rating.clamp(0, 5),
        a.complexity.clamp(0, 5),
        a.human_solved,
        json_str(&a.notes),
        a.excluded,
        a.updated
    )
}

fn annotations_json(key: &str, map: &HashMap<String, Annotation>) -> String {
    let mut ids: Vec<&String> = map.keys().collect();
    ids.sort();
    ids.retain(|id| map.get(*id).map_or(false, |a| !a.is_empty()));
    format!(
        "[{}]",
        ids.iter()
            .map(|id| annotation_json(key, id, &map[*id], false))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The same store in the on-disk JSONL form, for `GET /api/export`.
fn annotations_jsonl(key: &str, map: &HashMap<String, Annotation>) -> String {
    let mut ids: Vec<&String> = map.keys().collect();
    ids.sort();
    ids.retain(|id| map.get(*id).map_or(false, |a| !a.is_empty()));
    ids.iter()
        .map(|id| annotation_json(key, id, &map[*id], true) + "\n")
        .collect()
}

/// Make `human-unsolvable.jsonl` agree with a generated level's human verdict: drop the rows
/// this app wrote for these tokens, then, if `row_note` is given, add one row back (unless a
/// row someone else wrote already covers the level). Rows not written by this app are never
/// touched.
fn sync_human_unsolvable(
    path: &Path,
    tokens: &str,
    origin: &str,
    row_note: Option<&str>,
) -> Result<(), String> {
    let tokens = normalize_tokens(tokens);
    let old = std::fs::read_to_string(path).unwrap_or_default();
    // Other rows are kept byte for byte, line endings included; a new row matches them.
    let eol = if old.contains("\r\n") { "\r\n" } else { "\n" };
    let mut kept = String::new();
    let mut covered = false;
    let mut changed = false;
    for raw in old.split_inclusive('\n') {
        let line = raw.trim_end_matches(['\r', '\n']);
        if line.trim().is_empty() {
            continue;
        }
        let fields = object_fields(line);
        let same = field(&fields, "tokens")
            .map_or(false, |v| normalize_tokens(&json_unescape(v.trim_matches('"'))) == tokens);
        let ours = field(&fields, "source")
            .map_or(false, |v| json_unescape(v.trim_matches('"')).starts_with(HUMAN_SOURCE));
        if same && ours {
            changed = true;
            continue;
        }
        covered |= same;
        kept.push_str(raw);
        if !raw.ends_with('\n') {
            kept.push_str(eol);
        }
    }
    if let (Some(note), false) = (row_note, covered) {
        kept.push_str(&format!(
            "{{\"tokens\":{},\"verdict\":\"unsolvable\",\"source\":{},\"checked\":{},\"note\":{}}}{eol}",
            json_str(&tokens),
            json_str(&format!("{HUMAN_SOURCE} {origin}")),
            json_str(&today()),
            json_str(note)
        ));
        changed = true;
    }
    if !changed {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    write_atomic(path, &kept)
}

/// Today's UTC date as YYYY-MM-DD (days-to-civil, Howard Hinnant's algorithm).
fn today() -> String {
    let days = now_secs().div_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02}")
}

fn handle(mut stream: TcpStream, state: &Mutex<State>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    if reader.read_line(&mut request)? == 0 {
        return Ok(());
    }
    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" || header == "\n" {
            break;
        }
        if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = String::new();
    if content_length > 0 {
        reader
            .take(content_length as u64)
            .read_to_string(&mut body)?;
    }
    let mut words = request.split_whitespace();
    let method = words.next().unwrap_or("GET");
    let target = words.next().unwrap_or("/");
    let (path, query) = split_target(target);

    let reply = |stream: &mut TcpStream, ctype: &str, body: &str| -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            body.len()
        )?;
        stream.write_all(body.as_bytes())?;
        stream.flush()
    };

    match (method, path) {
        ("GET", "/") => reply(&mut stream, "text/html; charset=utf-8", APP)?,
        ("GET", "/api/config") => {
            let body = config_json(&state.lock().unwrap().paths);
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/levels") => {
            let body = state.lock().unwrap().levels_json.clone();
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/ratings") => {
            let body = state.lock().unwrap().ratings_json.clone();
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/items") => {
            let body = state.lock().unwrap().items_json.clone();
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/fetched") => {
            let body = state.lock().unwrap().fetched_json.clone();
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/annotations") => {
            let body = {
                let mut guard = state.lock().unwrap();
                // Agents may edit the JSONL file while the app is open; a reload shows their
                // changes on the next fetch or Rescan without restarting the server.
                let path = guard.paths.annotations.clone();
                if let Ok(annotations) = load_annotations(&path) {
                    guard.annotations = annotations;
                }
                annotations_json(guard.paths.id_key(), &guard.annotations)
            };
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/export") => {
            let body = {
                let mut guard = state.lock().unwrap();
                let path = guard.paths.annotations.clone();
                if let Ok(annotations) = load_annotations(&path) {
                    guard.annotations = annotations;
                }
                annotations_jsonl(guard.paths.id_key(), &guard.annotations)
            };
            reply(&mut stream, "application/x-ndjson", &body)?;
        }
        ("GET", "/api/level") => {
            let body = param(query, "id").map_or_else(
                || Err("missing ?id=".to_string()),
                |id| level_detail(state, &id),
            );
            match body {
                Ok(body) => reply(&mut stream, "application/json", &body)?,
                Err(why) => reply(&mut stream, "text/plain", &why)?,
            }
        }
        ("POST", "/api/annotate") => {
            let body = annotate(state, &body);
            match body {
                Ok(body) => reply(&mut stream, "application/json", &body)?,
                Err(why) => reply(&mut stream, "text/plain", &why)?,
            }
        }
        ("POST", "/api/stage") => {
            let body = match stage_request(state, &body) {
                Ok(body) => body,
                Err(why) => format!("{{\"ok\":false,\"error\":{}}}", json_str(&why)),
            };
            reply(&mut stream, "application/json", &body)?;
        }
        ("GET", "/api/rescan") => {
            let result = {
                let mut guard = state.lock().unwrap();
                guard.rescan().map(|()| guard.levels_json.clone())
            };
            match result {
                Ok(body) => reply(&mut stream, "application/json", &body)?,
                Err(why) => reply(&mut stream, "text/plain", &why)?,
            }
        }
        _ => {
            write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )?;
        }
    }
    Ok(())
}

fn level_detail(state: &Mutex<State>, id: &str) -> Result<String, String> {
    let state = state.lock().unwrap();
    let rec = state
        .index
        .get(id)
        .and_then(|&i| state.records.get(i))
        .ok_or_else(|| format!("no level {id}"))?;
    if rec.tokens.is_empty() {
        return Ok(format!(
            "{{\"id\":{},\"tokens\":\"\",\"geometry\":null,\"geometry_error\":\"no token sequence for this level\"}}",
            json_str(id)
        ));
    }
    let (geometry, error) = match tokens::decode(&rec.tokens) {
        Ok(puzzle) => {
            let level = Level::from_puzzle(&puzzle);
            (geometry_json(&level), String::new())
        }
        Err(why) => ("null".to_string(), why),
    };
    Ok(format!(
        "{{\"id\":{},\"tokens\":{},\"geometry\":{geometry},\"geometry_error\":{}}}",
        json_str(id),
        json_str(&rec.tokens),
        if error.is_empty() {
            "null".to_string()
        } else {
            json_str(&error)
        }
    ))
}

fn annotate(state: &Mutex<State>, body: &str) -> Result<String, String> {
    let id = param(body, "id").ok_or("missing id")?;
    let mut guard = state.lock().unwrap();
    let old = guard.annotations.get(&id).cloned().unwrap_or_default();
    let staged = param(body, "staged").map_or(old.staged, |v| v == "1");
    let rating = param(body, "rating")
        .map(|v| v.parse::<i64>().unwrap_or(old.rating))
        .unwrap_or(old.rating)
        .clamp(0, 5);
    let complexity = param(body, "complexity")
        .map(|v| v.parse::<i64>().unwrap_or(old.complexity))
        .unwrap_or(old.complexity)
        .clamp(0, 5);
    let human_solved = param(body, "human_solved").map_or(old.human_solved, |v| v == "1");
    let notes = param(body, "notes").unwrap_or_else(|| old.notes.clone());
    if notes.len() > 100_000 {
        return Err("notes are limited to 100,000 characters".into());
    }
    let excluded = param(body, "excluded").map_or(old.excluded, |v| v == "1");
    let human_verdict = param(body, "human_verdict").unwrap_or(old.human_verdict.clone());
    if !human_verdict.is_empty() && !HUMAN_VERDICTS.contains(&human_verdict.as_str()) {
        return Err(format!("unknown human verdict {human_verdict}"));
    }
    // A generated level's annotation keeps the level itself, so it outlives the run's files.
    let (tokens, origin) = match (&guard.paths.generated, guard.index.get(&id)) {
        (Some(_), Some(&i)) => {
            let r = &guard.records[i];
            (r.tokens.clone(), r.origin.clone())
        }
        _ => (old.tokens.clone(), old.origin.clone()),
    };
    let a = Annotation {
        staged,
        rating,
        complexity,
        human_solved,
        notes,
        excluded,
        human_verdict,
        tokens,
        origin,
        updated: now_secs(),
    };
    if let Some(g) = &guard.paths.generated {
        if (a.is_bad() || old.is_bad()) && !a.tokens.is_empty() {
            let note = a.is_bad().then(|| {
                let mut note = format!("Human play-test (workbench): {}.", a.human_verdict);
                if !a.notes.trim().is_empty() {
                    note.push(' ');
                    note.push_str(a.notes.trim());
                }
                note
            });
            sync_human_unsolvable(&g.human_unsolvable, &a.tokens, &a.origin, note.as_deref())?;
        }
    }
    let mut next = guard.annotations.clone();
    if a.is_empty() {
        next.remove(&id);
    } else {
        next.insert(id.clone(), a.clone());
    }
    save_annotations(&guard.paths.annotations, guard.paths.id_key(), &next)?;
    guard.annotations = next;
    Ok(annotation_json(guard.paths.id_key(), &id, &a, false))
}

/// What the page needs to know about the mode it is in.
fn config_json(paths: &Paths) -> String {
    let (mode, run, human) = match &paths.generated {
        Some(g) => (
            "generated",
            g.run.as_deref().map_or("null".to_string(), json_str),
            json_str(&g.human_unsolvable.display().to_string()),
        ),
        None => ("workshop", "null".to_string(), "null".to_string()),
    };
    format!(
        "{{\"mode\":{},\"run\":{run},\"annotations\":{},\"human_unsolvable\":{human}}}",
        json_str(mode),
        json_str(&paths.annotations.display().to_string())
    )
}

/// Stage one level with `tools/workshop/stage.py`. The script does the p2c compile, the copy
/// into the Puzzle Maker folder and the annotation update; this only serialises the call and
/// reloads the annotation store afterwards.
fn stage_request(state: &Mutex<State>, body: &str) -> Result<String, String> {
    let id = param(body, "id").ok_or("missing id")?;
    let force = param(body, "force").map_or(false, |v| v == "1");
    let paths = {
        let mut guard = state.lock().unwrap();
        if guard.staging {
            return Err("another level is being staged".into());
        }
        let i = *guard.index.get(&id).ok_or_else(|| format!("no level {id}"))?;
        let Some(rec) = guard.records.get(i).filter(|r| !r.tokens.is_empty()) else {
            return Err("this level has no token sequence to stage".into());
        };
        let level = guard.paths.generated.is_some().then(|| {
            let title = rec
                .fields
                .get("title")
                .map(|t| json_unescape(t.trim_matches('"')))
                .unwrap_or_else(|| id.clone());
            (rec.tokens.clone(), title, rec.origin.clone())
        });
        guard.staging = true;
        (guard.paths.clone(), level)
    };
    let (paths, level) = paths;
    let result = stage_level(&paths, &id, force, level);
    let mut guard = state.lock().unwrap();
    guard.staging = false;
    if result.is_ok() {
        let path = guard.paths.annotations.clone();
        if let Ok(annotations) = load_annotations(&path) {
            guard.annotations = annotations;
        }
    }
    result
}

/// `level` is a generated level's (tokens, title, origin): the tokens go to stage.py on stdin, since
/// the script can't look it up by id.
fn stage_level(
    paths: &Paths,
    id: &str,
    force: bool,
    level: Option<(String, String, String)>,
) -> Result<String, String> {
    let root = repo_root()?;
    let script = root.join("tools").join("workshop").join("stage.py");
    if !script.is_file() {
        return Err(format!("missing {}", script.display()));
    }
    let python = python_exe(&root);
    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg("--id")
        .arg(id)
        .arg("--annotations")
        .arg(&paths.annotations);
    match &level {
        Some((_, title, origin)) => {
            cmd.args(["--tokens-stdin", "--key", "id", "--series", "gen", "--title", title])
                .args(["--origin", origin])
                .arg("--registry")
                .arg(DEFAULT_GEN_REGISTRY);
        }
        None => {
            cmd.arg("--all")
                .arg(&paths.all)
                .arg("--train")
                .arg(&paths.train)
                .arg("--test")
                .arg(&paths.test);
        }
    }
    if force {
        cmd.arg("--force");
    }
    let out = match &level {
        Some((tokens, _, _)) => {
            cmd.stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            let mut child = cmd
                .spawn()
                .map_err(|e| format!("can't run {}: {e}", python.display()))?;
            if let Some(mut stdin) = child.stdin.take() {
                stdin
                    .write_all(tokens.as_bytes())
                    .map_err(|e| format!("stage.py stdin: {e}"))?;
            }
            child.wait_with_output()
        }
        None => cmd.output(),
    }
    .map_err(|e| format!("can't run {}: {e}", python.display()))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .map(str::trim);
    if !out.status.success() {
        // stage.py prints a JSON error object on stdout even when it exits non-zero.
        if let Some(json) = json {
            return Ok(json.to_string());
        }
        return Err(format!("staging failed: {}", tail_chars(&stderr, 2000)));
    }
    json.map(str::to_string).ok_or_else(|| {
        let detail = tail_chars(&stderr, 1000);
        if detail.is_empty() {
            "stage.py printed no result".into()
        } else {
            format!("stage.py printed no result: {detail}")
        }
    })
}

fn repo_root() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("current directory: {e}"))?;
    if cwd.join("tools").join("workshop").join("stage.py").is_file() {
        return Ok(cwd);
    }
    let exe = std::env::current_exe().map_err(|e| format!("current executable: {e}"))?;
    let root = exe
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or("can't find the repo root from the glados executable")?;
    if root.join("tools").join("workshop").join("stage.py").is_file() {
        return Ok(root.to_path_buf());
    }
    Err("can't find tools/workshop/stage.py; run glados-workbench from the repo root".into())
}

fn python_exe(root: &Path) -> PathBuf {
    for path in [
        root.join(".venv").join("Scripts").join("python.exe"),
        root.join(".venv").join("bin").join("python"),
    ] {
        if path.is_file() {
            return path;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn tail_chars(s: &str, n: usize) -> String {
    let mut chars: Vec<char> = s.trim().chars().rev().take(n).collect();
    chars.reverse();
    chars.into_iter().collect()
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn split_target(target: &str) -> (&str, &str) {
    match target.split_once('?') {
        Some((path, query)) => (path, query),
        None => (target, ""),
    }
}

fn param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = (bytes[i + 1] as char).to_digit(16);
                let lo = (bytes[i + 2] as char).to_digit(16);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---- a very small JSON reader for our JSONL ----------------------------------------------
//
// The files are written by Python's `json.dumps`/our `json_str`: one object per line, with no
// comments. We only need a line's top-level keys and their raw values, plus unescaping for
// ids and notes. Keeping the values verbatim means a field added tomorrow still reaches the
// page without changing this code.

fn field<'a>(fields: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    fields
        .iter()
        .rev()
        .find_map(|&(k, v)| (k == key).then_some(v))
}

fn string_end(b: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

fn value_end(b: &[u8], start: usize) -> Option<usize> {
    if start >= b.len() {
        return None;
    }
    match b[start] {
        b'"' => string_end(b, start),
        open @ (b'{' | b'[') => {
            let close = if open == b'{' { b'}' } else { b']' };
            let mut depth = 0usize;
            let mut i = start;
            while i < b.len() {
                match b[i] {
                    b'"' => {
                        i = string_end(b, i)?;
                        continue;
                    }
                    c if c == open => depth += 1,
                    c if c == close => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(i + 1);
                        }
                    }
                    _ => {}
                }
                i += 1;
            }
            None
        }
        _ => {
            let mut i = start;
            while i < b.len() && b[i] != b',' && b[i] != b'}' && b[i] != b']' {
                i += 1;
            }
            Some(i)
        }
    }
}

/// Top-level `(key, raw value)` pairs of a JSON object.
fn object_fields(line: &str) -> Vec<(&str, &str)> {
    let b = line.as_bytes();
    let Some(start) = b.iter().position(|&c| c == b'{') else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = start + 1;
    loop {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b',') {
            i += 1;
        }
        if i >= b.len() || b[i] == b'}' {
            break;
        }
        if b[i] != b'"' {
            break;
        }
        let Some(key_end) = string_end(b, i) else {
            break;
        };
        let key = &line[i + 1..key_end - 1];
        i = key_end;
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() || b[i] != b':' {
            break;
        }
        i += 1;
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let Some(end) = value_end(b, i) else {
            break;
        };
        out.push((key, &line[i..end]));
        i = end;
    }
    out
}

/// Undo JSON string escapes (the source may be a whole quoted string or already unquoted).
pub(crate) fn json_unescape(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(&e) = chars.get(i) else {
            break;
        };
        i += 1;
        match e {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            '/' => out.push('/'),
            'b' => out.push('\u{8}'),
            'f' => out.push('\u{c}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                let Some(hi) = hex4(&chars, i) else {
                    out.push('\u{fffd}');
                    continue;
                };
                i += 4;
                let code = if (0xd800..0xdc00).contains(&hi) && chars.get(i) == Some(&'\\') {
                    if chars.get(i + 1) == Some(&'u') {
                        if let Some(lo) = hex4(&chars, i + 2) {
                            i += 6;
                            0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00)
                        } else {
                            hi
                        }
                    } else {
                        hi
                    }
                } else {
                    hi
                };
                out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
            }
            other => out.push(other),
        }
    }
    out
}

fn hex4(chars: &[char], i: usize) -> Option<u32> {
    let mut n = 0u32;
    for k in 0..4 {
        n = n * 16 + chars.get(i + k)?.to_digit(16)?;
    }
    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_fields_keep_nested_values_verbatim() {
        let line = r#"{"workshop_id":"12","title":"a \"quoted\" name","metrics":{"score":3.5,"bits":[1,2]}, "verdict":"solvable"}"#;
        let f = object_fields(line);
        assert_eq!(field(&f, "workshop_id"), Some(r#""12""#));
        assert_eq!(field(&f, "title"), Some(r#""a \"quoted\" name""#));
        assert_eq!(field(&f, "metrics"), Some(r#"{"score":3.5,"bits":[1,2]}"#));
        assert_eq!(field(&f, "verdict"), Some(r#""solvable""#));
    }

    #[test]
    fn json_unescape_handles_quotes_and_newlines() {
        assert_eq!(json_unescape(r#"a \"b\"\nnext"#), "a \"b\"\nnext");
        assert_eq!(json_unescape(r#"\u00e9"#), "é");
        assert_eq!(json_unescape(r#"\ud83d\ude00"#), "😀");
    }

    #[test]
    fn annotations_round_trip() {
        let dir = std::env::temp_dir().join(format!("glados-wb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notes.jsonl");
        let mut map = HashMap::new();
        map.insert(
            "42".to_string(),
            Annotation {
                staged: true,
                rating: 4,
                complexity: 3,
                human_solved: true,
                notes: "a \"note\"\nwith a newline".into(),
                excluded: true,
                updated: 7,
                ..Annotation::default()
            },
        );
        map.insert("empty".into(), Annotation::default());
        save_annotations(&path, "workshop_id", &map).unwrap();
        // The Workshop store keeps its old shape: no generated-level fields.
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("human_verdict") && !text.contains("tokens"), "{text}");
        let back = load_annotations(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back["42"].notes, "a \"note\"\nwith a newline");
        assert!(back["42"].staged && back["42"].excluded);
        assert_eq!(back["42"].rating, 4);
        assert_eq!(back["42"].complexity, 3);
        assert!(back["42"].human_solved);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn records_merge_without_losing_base_fields() {
        let dir = std::env::temp_dir().join(format!("glados-wb-merge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let all = dir.join("all.jsonl");
        let train = dir.join("train.jsonl");
        let test = dir.join("test.jsonl");
        std::fs::write(
            &all,
            "{\"workshop_id\":\"1\",\"verdict\":\"unsolvable\",\"metrics\":null,\"tokens\":\"size 1 1 1\"}\n",
        )
        .unwrap();
        std::fs::write(
            &train,
            "{\"workshop_id\":\"1\",\"verdict\":\"published\",\"solver_verdict\":\"unsolvable\",\"tokens\":\"size 1 1 1\"}\n",
        )
        .unwrap();
        std::fs::write(&test, "").unwrap();
        let records = load_levels(&Paths {
            all: all.clone(),
            train: train.clone(),
            test: test.clone(),
            ..Paths::default()
        })
        .unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].split, Some("train"));
        assert_eq!(records[0].fields["verdict"], "\"published\"");
        assert_eq!(records[0].fields["solver_verdict"], "\"unsolvable\"");
        assert!(records[0].fields.contains_key("metrics"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generated_levels_merge_by_tokens() {
        let dir = std::env::temp_dir().join(format!("glados-wb-gen-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let run = dir.join("runs").join("r1");
        std::fs::create_dir_all(run.join("eval").join("ingame")).unwrap();
        std::fs::write(
            run.join("eval").join("samples.jsonl"),
            concat!(
                "{\"line\": 1, \"tokens\": \"size 1 1 1\", \"verdict\": \"solvable\", \"setting\": \"baseline\", \"sample\": \"x\"}\n",
                "{\"line\": 2, \"tokens\": \"\", \"verdict\": \"unparsed\"}\n",
            ),
        )
        .unwrap();
        std::fs::write(
            run.join("eval").join("rated-samples.jsonl"),
            "{\"i\": 0, \"line\": 5, \"tokens\": \"size  1 1 1\", \"outcome\": \"uncertain\", \"novel\": true}\n",
        )
        .unwrap();
        std::fs::write(
            run.join("eval").join("ingame").join("results.jsonl"),
            "{\"tokens\": \"size 1 1 1\", \"game\": {\"verdict\": \"valid\"}}\n",
        )
        .unwrap();
        let g = Generated {
            run: Some("r1".into()),
            run_dir: Some(run.clone()),
            extra: Vec::new(),
            human_unsolvable: dir.join("human.jsonl"),
        };
        let records = load_generated_levels(&g).unwrap();
        assert_eq!(records.len(), 1, "the unparsed line is skipped, the copies merge");
        let r = &records[0];
        assert_eq!(r.id, token_id("size 1 1 1"));
        assert_eq!(r.sources, vec!["eval".to_string(), "rated".to_string()]);
        assert_eq!(r.fields["solver_verdict"], "\"solvable\"");
        assert_eq!(r.fields["novel"], "true");
        assert_eq!(r.fields["game"], "{\"verdict\": \"valid\"}");
        assert!(!r.fields.contains_key("sample"));
        assert_eq!(r.origin, "r1 eval line 1");
        assert_eq!(newest_level_run(&dir.join("runs")), None, "no status.json yet");
        std::fs::write(run.join("status.json"), "{\n  \"started\": \"2026-10-26T10:47:17\",\n  \"task\": \"levels\"\n}").unwrap();
        assert_eq!(newest_level_run(&dir.join("runs")), Some(run.clone()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn human_unsolvable_rows_come_and_go() {
        let dir = std::env::temp_dir().join(format!("glados-wb-human-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("human-unsolvable.jsonl");
        // CRLF, like the real file: rows we don't touch must come back byte for byte.
        let foreign = "{\"tokens\": \"size 2 2 2\", \"verdict\": \"unsolvable\", \"source\": \"runs/x\"}\r\n";
        std::fs::write(&path, foreign).unwrap();
        sync_human_unsolvable(&path, "size 1 1 1", "r1 eval line 1", Some("broken")).unwrap();
        sync_human_unsolvable(&path, "size 1  1 1", "r1 eval line 1", Some("broken, again")).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2, "{text}");
        assert!(text.contains("\"source\":\"glados-workbench r1 eval line 1\""));
        assert!(text.contains("broken, again") && text.contains(&format!("\"checked\":\"{}\"", today())));
        // A level someone else already recorded gets no second row, and keeps theirs.
        sync_human_unsolvable(&path, "size 2 2 2", "r1 eval line 9", Some("dup")).unwrap();
        sync_human_unsolvable(&path, "size 2 2 2", "r1 eval line 9", None).unwrap();
        sync_human_unsolvable(&path, "size 1 1 1", "", None).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), foreign);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn today_is_a_date() {
        let t = today();
        assert_eq!(t.len(), 10);
        assert!(t.as_str() >= "2026-01-01" && &t[4..5] == "-", "{t}");
    }

    #[test]
    fn detail_decodes_the_tokens() {
        let puzzle = crate::fixtures::flat_room();
        let text = tokens::encode(&puzzle).unwrap().join(" ");
        let state = Mutex::new(State {
            records: vec![Record {
                id: "1".into(),
                fields: BTreeMap::new(),
                tokens: text.clone(),
                split: None,
                sources: vec!["workshop".into()],
                origin: String::new(),
            }],
            index: HashMap::from([("1".to_string(), 0)]),
            levels_json: String::new(),
            ratings_json: "{}".into(),
            items_json: "{}".into(),
            fetched_json: "{}".into(),
            annotations: HashMap::new(),
            paths: Paths::default(),
            staging: false,
        });
        let out = level_detail(&state, "1").unwrap();
        assert!(out.contains("\"geometry\":{"), "{out}");
        assert!(out.contains("\"size\":"), "{out}");
        assert!(!out.contains("\"geometry_error\":\""));
    }
}
