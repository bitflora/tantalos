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
//! Endpoints:
//! - `GET /`                    the app
//! - `GET /api/levels`          every level, merged across the corpus and the ws5 splits
//! - `GET /api/ratings`         Workshop `{id: {stars, votes}}`
//! - `GET /api/items`           Workshop item metadata (`items.jsonl`) by id
//! - `GET /api/fetched`         download status (`fetched.jsonl`) by id
//! - `GET /api/annotations`     our annotations, one object per level
//! - `GET /api/level?id=ID`     one level's tokens and decoded geometry
//! - `POST /api/annotate`       update one annotation (form fields: id, staged, rating,
//!                              notes, excluded)
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
                         out/workshop/{ratings,items,fetched}.jsonl)";

const DEFAULT_ALL: &str = "out/workshop/levels.jsonl";
const DEFAULT_TRAIN: &str = "data/ws5/train.jsonl";
const DEFAULT_TEST: &str = "data/ws5/test.jsonl";
const DEFAULT_RATINGS: &str = "out/workshop/ratings.jsonl";
const DEFAULT_ITEMS: &str = "out/workshop/items.jsonl";
const DEFAULT_FETCHED: &str = "out/workshop/fetched.jsonl";
const DEFAULT_ANNOTATIONS: &str = "train/workshop-annotations.jsonl";

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
    sources: Vec<&'static str>,
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
    pub updated: i64,
}

impl Annotation {
    fn is_empty(&self) -> bool {
        !self.staged && self.rating == 0 && self.complexity == 0 && !self.human_solved && self.notes.is_empty() && !self.excluded
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
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = |it: &mut std::slice::Iter<String>| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--all" | "--levels" => paths.all = PathBuf::from(value(&mut it)?),
            "--train" => paths.train = PathBuf::from(value(&mut it)?),
            "--test" => paths.test = PathBuf::from(value(&mut it)?),
            "--ratings" => paths.ratings = PathBuf::from(value(&mut it)?),
            "--items" => paths.items = PathBuf::from(value(&mut it)?),
            "--fetched" => paths.fetched = PathBuf::from(value(&mut it)?),
            "--annotations" => paths.annotations = PathBuf::from(value(&mut it)?),
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
    println!(
        "glados-workbench: {} levels ({by_verdict})",
        state.records.len()
    );
    println!(
        "  {} annotations ({staged} staged, {excluded} excluded) -> {}",
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
        let records = load_levels(paths)?;
        if records.is_empty() {
            return Err(format!(
                "no levels found: expected {} (or pass --all/--train/--test)",
                paths.all.display()
            ));
        }
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
            ratings_json: load_ratings(&paths.ratings),
            items_json: load_map(&paths.items, "id"),
            fetched_json: load_map(&paths.fetched, "id"),
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
        if !rec.sources.contains(&source) {
            rec.sources.push(source);
        }
    }
    Ok(())
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
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}: {e}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields = object_fields(&line);
        let Some(raw_id) = field(&fields, "workshop_id") else {
            return Err(format!("{}:{}: no workshop_id", path.display(), n + 1));
        };
        let id = json_unescape(raw_id.trim_matches('"'));
        let notes = field(&fields, "notes")
            .map(|v| json_unescape(v.trim_matches('"')))
            .unwrap_or_default();
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
                notes,
                excluded: field(&fields, "excluded").map_or(false, |v| v == "true"),
                updated: field(&fields, "updated")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0),
            },
        );
    }
    Ok(out)
}

fn save_annotations(path: &Path, map: &HashMap<String, Annotation>) -> Result<(), String> {
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
        text.push_str(&format!(
            "{{\"workshop_id\":{},\"staged\":{},\"rating\":{},\"complexity\":{},\"human_solved\":{},\"notes\":{},\"excluded\":{},\"updated\":{}}}\n",
            json_str(id),
            a.staged,
            a.rating.clamp(0, 5),
            a.complexity.clamp(0, 5),
            a.human_solved,
            json_str(&a.notes),
            a.excluded,
            a.updated
        ));
    }
    let tmp = path.with_extension("jsonl.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

fn annotation_json(id: &str, a: &Annotation) -> String {
    format!(
        "{{\"workshop_id\":{},\"staged\":{},\"rating\":{},\"complexity\":{},\"human_solved\":{},\"notes\":{},\"excluded\":{},\"updated\":{}}}",
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

fn annotations_json(map: &HashMap<String, Annotation>) -> String {
    let mut ids: Vec<&String> = map.keys().collect();
    ids.sort();
    ids.retain(|id| map.get(*id).map_or(false, |a| !a.is_empty()));
    format!(
        "[{}]",
        ids.iter()
            .map(|id| annotation_json(id, &map[*id]))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// The same store in the on-disk JSONL form, for `GET /api/export`.
fn annotations_jsonl(map: &HashMap<String, Annotation>) -> String {
    let mut ids: Vec<&String> = map.keys().collect();
    ids.sort();
    ids.retain(|id| map.get(*id).map_or(false, |a| !a.is_empty()));
    ids.iter()
        .map(|id| annotation_json(id, &map[*id]) + "\n")
        .collect()
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
                annotations_json(&guard.annotations)
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
                annotations_jsonl(&guard.annotations)
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
    let notes = param(body, "notes").unwrap_or(old.notes);
    if notes.len() > 100_000 {
        return Err("notes are limited to 100,000 characters".into());
    }
    let excluded = param(body, "excluded").map_or(old.excluded, |v| v == "1");
    let a = Annotation {
        staged,
        rating,
        complexity,
        human_solved,
        notes,
        excluded,
        updated: now_secs(),
    };
    let mut next = guard.annotations.clone();
    if a.is_empty() {
        next.remove(&id);
    } else {
        next.insert(id.clone(), a.clone());
    }
    save_annotations(&guard.paths.annotations, &next)?;
    guard.annotations = next;
    Ok(annotation_json(&id, &a))
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
        if guard.records.get(i).map_or(true, |r| r.tokens.is_empty()) {
            return Err("this level has no token sequence to stage".into());
        }
        guard.staging = true;
        guard.paths.clone()
    };
    let result = stage_level(&paths, &id, force);
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

fn stage_level(paths: &Paths, id: &str, force: bool) -> Result<String, String> {
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
        .arg("--all")
        .arg(&paths.all)
        .arg("--train")
        .arg(&paths.train)
        .arg("--test")
        .arg(&paths.test)
        .arg("--annotations")
        .arg(&paths.annotations);
    if force {
        cmd.arg("--force");
    }
    let out = cmd
        .output()
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
            },
        );
        map.insert("empty".into(), Annotation::default());
        save_annotations(&path, &map).unwrap();
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
    fn detail_decodes_the_tokens() {
        let puzzle = crate::fixtures::flat_room();
        let text = tokens::encode(&puzzle).unwrap().join(" ");
        let state = Mutex::new(State {
            records: vec![Record {
                id: "1".into(),
                fields: BTreeMap::new(),
                tokens: text.clone(),
                split: None,
                sources: vec!["workshop"],
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
