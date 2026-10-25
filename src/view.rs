//! A graphical viewer for the solver's solutions.
//!
//! `glados-view` scans folders for saved routes — `<name>.route.jsonl` files with their
//! `<name>.p2c` level beside them, exactly what `glados-score route` (redirected) and
//! `glados-score check --route DIR` write — serves the app and those levels over a local port, and
//! opens the browser. Picking a solution loads its geometry and route; nothing is re-solved, so
//! switching levels is instant.
//!
//! The page (`viewer.html`, embedded) draws the chamber in oblique 2.5D (Z up) and plays the
//! route back: the walks, the portals, the cube and the player, driven by the `after` state of
//! each leg. It works the same for a plan the Rust solver made and for one the solver net wrote.
//!
//! Endpoints:
//! - `GET /`                    the app (its data placeholder is `null`; it lists from the API)
//! - `GET /api/items`           every solution found, grouped by folder
//! - `GET /api/item?path=P`     one solution's geometry (from the `.p2c`) and route (the JSONL)
//! - `GET /api/rescan`          re-scan the roots, then return the items
//! - `GET /api/standalone?path=P`  one solution inlined into a shareable, self-contained page

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::dataset::json_str;
use crate::geom::{step, Cell, Dir};
use crate::level::{
    BarrierKind, ButtonKind, HazardKind, ItemKind, Level, PaintKind, PanelKind, ProjectorKind,
};
use crate::p2c::Puzzle;

/// The page, with `__GLADOS_DATA__` where the data goes (or `null`, in server mode).
const APP: &str = include_str!("viewer.html");
const PLACEHOLDER: &str = "__GLADOS_DATA__";

/// A pickable solution: a saved route and the level beside it.
pub struct Found {
    pub route: PathBuf,
    pub level: PathBuf,
    /// The sidebar group: the scan root's name, plus the folder under it (uses `/`).
    pub group: String,
    pub title: String,
    pub size: [i32; 3],
    /// Legs in the route, from the file's header.
    pub legs: usize,
}

/// Every `.route.jsonl` under `roots` that has a `.p2c` beside it.
pub fn scan(roots: &[PathBuf]) -> Vec<Found> {
    let mut out = Vec::new();
    for root in roots {
        walk(root, root, &mut out);
    }
    out.sort_by(|a, b| {
        a.group
            .cmp(&b.group)
            .then_with(|| a.title.cmp(&b.title))
            .then_with(|| a.route.cmp(&b.route))
    });
    out
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<Found>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out);
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let Some(stem) = name.strip_suffix(".route.jsonl") else {
            continue;
        };
        let level = path.with_file_name(format!("{stem}.p2c"));
        if !level.is_file() {
            continue;
        }
        let Ok(header) = first_line(&path) else {
            continue;
        };
        let Some((title, size, legs)) = header_fields(&header) else {
            continue;
        };
        // Group by the root's name plus the folder it sits in, so solutions from different
        // roots (and different sub-folders) stay apart in the sidebar.
        let base = root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| root.display().to_string());
        let under = path
            .parent()
            .and_then(|d| d.strip_prefix(root).ok())
            .map(|d| d.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();
        let group = if under.is_empty() {
            base
        } else {
            format!("{base}/{under}")
        };
        out.push(Found {
            route: path,
            level,
            group,
            title,
            size,
            legs,
        });
    }
}

fn first_line(path: &Path) -> std::io::Result<String> {
    let mut line = String::new();
    BufReader::new(std::fs::File::open(path)?).read_line(&mut line)?;
    Ok(line)
}

/// `title`, `size` and `legs` out of a route's header line (a JSON object we wrote).
fn header_fields(line: &str) -> Option<(String, [i32; 3], usize)> {
    Some((
        json_string_field(line, "title")?,
        json_ints_field(line, "size")?,
        json_usize_field(line, "legs")?,
    ))
}

fn json_string_field(line: &str, key: &str) -> Option<String> {
    let rest = line
        .split_once(&format!("\"{key}\":"))?
        .1
        .strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                other => out.push(other),
            },
            other => out.push(other),
        }
    }
    None
}

fn json_ints_field(line: &str, key: &str) -> Option<[i32; 3]> {
    let rest = line.split_once(&format!("\"{key}\":["))?.1;
    let end = rest.find(']')?;
    let nums: Vec<i32> = rest[..end]
        .split(',')
        .filter_map(|n| n.trim().parse().ok())
        .collect();
    (nums.len() == 3).then(|| [nums[0], nums[1], nums[2]])
}

fn json_usize_field(line: &str, key: &str) -> Option<usize> {
    let rest = line.split_once(&format!("\"{key}\":"))?.1;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Serve the app and the found solutions on `127.0.0.1:port` (port 0: any free one).
pub fn serve(roots: Vec<PathBuf>, port: u16, open: bool) -> Result<(), String> {
    let found = Arc::new(Mutex::new(scan(&roots)));
    let roots = Arc::new(roots);
    let listener = TcpListener::bind(("127.0.0.1", port))
        .map_err(|e| format!("can't listen on 127.0.0.1:{port}: {e}"))?;
    let actual = listener.local_addr().map_err(|e| e.to_string())?.port();
    let n = found.lock().unwrap().len();
    println!(
        "glados-view: {n} solutions under {}",
        roots
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if n == 0 {
        println!("  (no <name>.route.jsonl + <name>.p2c pairs found; make some with `glados-score route` or `glados-score check --route`)");
    }
    let url = format!("http://127.0.0.1:{actual}/");
    println!("  {url}   (Ctrl-C to stop)");
    if open {
        open_browser(&url);
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else {
            continue;
        };
        let found = Arc::clone(&found);
        let roots = Arc::clone(&roots);
        std::thread::spawn(move || {
            let _ = handle(stream, &found, &roots);
        });
    }
    Ok(())
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

fn handle(
    mut stream: TcpStream,
    found: &Mutex<Vec<Found>>,
    roots: &[PathBuf],
) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    if reader.read_line(&mut request)? == 0 {
        return Ok(());
    }
    // Read and drop the headers.
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 || header == "\r\n" || header == "\n" {
            break;
        }
    }
    let target = request.split_whitespace().nth(1).unwrap_or("/").to_string();
    let (path, query) = split_target(&target);

    let reply = |stream: &mut TcpStream, ctype: &str, body: &str| -> std::io::Result<()> {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            body.len()
        )?;
        stream.write_all(body.as_bytes())?;
        stream.flush()
    };

    match path {
        "/" => reply(
            &mut stream,
            "text/html; charset=utf-8",
            &APP.replace(PLACEHOLDER, "null"),
        )?,
        "/api/items" => {
            let body = items_json(&found.lock().unwrap());
            reply(&mut stream, "application/json", &body)?;
        }
        "/api/rescan" => {
            *found.lock().unwrap() = scan(roots);
            let body = items_json(&found.lock().unwrap());
            reply(&mut stream, "application/json", &body)?;
        }
        "/api/item" => {
            let body = param(query, "path").map_or_else(
                || Err("missing ?path=".to_string()),
                |p| item_json(found, &p),
            );
            match body {
                Ok(body) => reply(&mut stream, "application/json", &body)?,
                Err(why) => reply(&mut stream, "text/plain", &why)?,
            }
        }
        "/api/standalone" => {
            let body = param(query, "path").map_or_else(
                || Err("missing ?path=".to_string()),
                |p| item_json(found, &p),
            );
            match body {
                Ok(body) => {
                    let body = body
                        .strip_suffix('}')
                        .map_or(body.clone(), |b| format!("{b},\"standalone\":true}}"));
                    let page = APP.replace(PLACEHOLDER, &body);
                    reply(&mut stream, "text/html; charset=utf-8", &page)?;
                }
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

fn items_json(found: &[Found]) -> String {
    let items: Vec<String> = found
        .iter()
        .map(|f| {
            format!(
                "{{\"path\":{},\"group\":{},\"title\":{},\"size\":[{},{},{}],\"legs\":{}}}",
                json_str(&f.route.display().to_string()),
                json_str(&f.group),
                json_str(&f.title),
                f.size[0],
                f.size[1],
                f.size[2],
                f.legs
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// The geometry (from the level) and route (from the JSONL) of the solution at `path`.
fn item_json(found: &Mutex<Vec<Found>>, path: &str) -> Result<String, String> {
    let wanted = PathBuf::from(path);
    let (route_path, level_path, name) = {
        let found = found.lock().unwrap();
        let f = found
            .iter()
            .find(|f| f.route == wanted)
            .ok_or("not a listed solution")?;
        let name = f
            .route
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .trim_end_matches(".jsonl")
            .trim_end_matches(".route")
            .to_string();
        (f.route.clone(), f.level.clone(), name)
    };
    let source = std::fs::read_to_string(&level_path)
        .map_err(|e| format!("{}: {e}", level_path.display()))?;
    let puzzle = Puzzle::parse(&source).map_err(|e| format!("{}: {e}", level_path.display()))?;
    let level = Level::from_puzzle(&puzzle);
    let geometry = geometry_json(&level);

    let route_text = std::fs::read_to_string(&route_path)
        .map_err(|e| format!("{}: {e}", route_path.display()))?;
    let lines: Vec<&str> = route_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect();
    if lines.is_empty() {
        return Err("the route file is empty".into());
    }
    let legs = lines[1..].join(",");
    let route = format!("{{\"header\":{},\"legs\":[{legs}]}}", lines[0]);

    // Keep the name out of the JSON unless it adds something over the title in the header.
    Ok(format!(
        "{{\"name\":{},\"geometry\":{geometry},\"route\":{route}}}",
        json_str(&name)
    ))
}

pub(crate) fn geometry_json(level: &Level) -> String {
    let mut solid = Vec::new();
    let mut goo = Vec::new();
    for c in level.cells() {
        if level.solid(c) {
            solid.push(cell_json(c));
        } else if level.goo(c) {
            goo.push(cell_json(c));
        }
    }
    let white: Vec<String> = level
        .portal_surfaces()
        .iter()
        .map(|&(c, d)| format!("{},\"{}\"", cell_json(c), dir_name(d)))
        .collect();

    let mut items = Vec::new();
    for item in &level.items {
        if let Some(json) = item_json_marker(item) {
            items.push(json);
        }
    }
    // Turrets, lasers, plates, projectors, panels, stairs, pistons and rails carry data the raw
    // items don't (a turret's yaw, a field's direction, a rail's track); the structured lists
    // have it.
    for t in &level.turrets {
        items.push(format!(
            "{{\"k\":\"turret\",\"p\":{},\"d\":{:.1},\"i\":{}}}",
            cell_json(t.cell),
            t.yaw,
            t.item
        ));
    }
    for l in &level.lasers {
        let k = match l.kind {
            crate::level::LaserKind::Emitter => "laser",
            crate::level::LaserKind::Catcher => "catcher",
            crate::level::LaserKind::Relay => "relay",
        };
        items.push(format!(
            "{{\"k\":\"{k}\",\"p\":{},\"d\":\"{}\",\"i\":{}}}",
            cell_json(l.cell),
            dir_name(l.normal),
            l.item
        ));
    }
    for p in &level.plates {
        items.push(format!(
            "{{\"k\":\"plate\",\"p\":{},\"to\":{},\"i\":{}}}",
            cell_json(p.cell),
            cell_json(p.target.1),
            p.item
        ));
    }
    for p in &level.projectors {
        let k = match p.kind {
            ProjectorKind::Bridge => "bridge",
            ProjectorKind::Funnel => "funnel",
        };
        items.push(format!(
            "{{\"k\":\"{k}\",\"p\":{},\"d\":\"{}\",\"i\":{}}}",
            cell_json(p.cell),
            dir_name(p.normal),
            p.item
        ));
    }
    for p in &level.panels {
        let k = match p.kind {
            PanelKind::Angled => "panel",
            PanelKind::Glass => "glasspanel",
            PanelKind::Flip => "flippanel",
        };
        items.push(format!(
            "{{\"k\":\"{k}\",\"p\":{},\"d\":\"{}\",\"tilt\":\"{}\",\"angle\":{},\"i\":{}}}",
            cell_json(p.cell),
            dir_name(p.normal),
            dir_name(p.tilt),
            p.angle,
            p.item
        ));
    }
    for s in &level.stairs {
        items.push(format!(
            "{{\"k\":\"stairs\",\"p\":{},\"d\":\"{}\",\"i\":{}}}",
            cell_json(s.cell),
            dir_name(s.dir),
            s.item
        ));
    }
    for p in &level.pistons {
        items.push(format!(
            "{{\"k\":\"piston\",\"p\":{},\"d\":\"{}\",\"i\":{}}}",
            cell_json(p.cell),
            dir_name(p.normal),
            p.item
        ));
    }
    for r in &level.rails {
        let track: Vec<String> = r.track().iter().map(|&c| cell_json(c)).collect();
        items.push(format!(
            "{{\"k\":\"rail\",\"p\":{},\"track\":[{}],\"i\":{}}}",
            cell_json(r.cell),
            track.join(","),
            r.item
        ));
    }

    let barriers: Vec<String> = level
        .barriers
        .iter()
        .map(|b| {
            let kind = match b.kind {
                BarrierKind::Glass => "glass",
                BarrierKind::Grating => "grating",
            };
            format!(
                "{{\"kind\":\"{kind}\",\"faces\":[{}]}}",
                faces_json(&b.faces)
            )
        })
        .collect();
    let hazards: Vec<String> = level
        .hazards
        .iter()
        .map(|h| {
            let kind = match h.kind {
                HazardKind::Fizzler => "fizzler",
                HazardKind::LaserField => "laserfield",
            };
            format!(
                "{{\"kind\":\"{kind}\",\"on\":{},\"faces\":[{}]}}",
                h.start_enabled,
                faces_json(&h.faces)
            )
        })
        .collect();
    let paint: Vec<String> = level
        .gels
        .iter()
        .map(|g| {
            let kind = match g.kind {
                PaintKind::Bounce => "bounce",
                PaintKind::Speed => "speed",
                PaintKind::Portal => "portal",
                PaintKind::Water => "water",
            };
            let dropper = g.dropper.map_or("null".to_string(), |(_, c)| cell_json(c));
            format!(
                "{{\"kind\":\"{kind}\",\"face\":[{},\"{}\"],\"dropper\":{dropper}}}",
                cell_json(g.face.0),
                dir_name(g.face.1)
            )
        })
        .collect();
    let links: Vec<String> = level
        .links
        .iter()
        .map(|l| format!("[{},{}]", l.sender, l.receiver))
        .collect();
    let beams = beams_json(level);

    format!(
        "{{\"size\":[{},{},{}],\"solid\":[{}],\"goo\":[{}],\"white\":[{}],\"items\":[{}],\"barriers\":[{}],\"hazards\":[{}],\"paint\":[{}],\"links\":[{}],\"beams\":{beams}}}",
        level.size[0],
        level.size[1],
        level.size[2],
        solid.join(","),
        goo.join(","),
        white.join(","),
        items.join(","),
        barriers.join(","),
        hazards.join(","),
        paint.join(","),
        links.join(",")
    )
}

/// The emitters' beams, as polylines in voxel coordinates: the layout as it starts, before any
/// portal or reflection cube redirects them. Enough to see what a laser level is doing.
fn beams_json(level: &Level) -> String {
    let mut out = Vec::new();
    for l in level
        .lasers
        .iter()
        .filter(|l| l.kind == crate::level::LaserKind::Emitter)
    {
        let mut cell = l.cell;
        let mut points = vec![centre(cell)];
        for _ in 0..128 {
            let next = step(cell, l.normal);
            if level.solid(next) {
                points.push(face_centre(cell, l.normal));
                break;
            }
            cell = next;
            points.push(centre(cell));
        }
        let line: Vec<String> = points.iter().map(|p| point_json(*p)).collect();
        out.push(format!("[{}]", line.join(",")));
    }
    format!("[{}]", out.join(","))
}

/// The centre of a cell's voxel.
fn centre(c: Cell) -> [f64; 3] {
    [c[0] as f64 + 0.5, c[1] as f64 + 0.5, c[2] as f64 + 0.5]
}

/// The centre of the face of `c` on the `d` side.
fn face_centre(c: Cell, d: Dir) -> [f64; 3] {
    let v = d.vector();
    let mut p = centre(c);
    for (i, s) in v.iter().enumerate() {
        p[i] += *s as f64 * 0.5;
    }
    p
}

fn point_json(p: [f64; 3]) -> String {
    let n = |v: f64| {
        let s = format!("{v:.3}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s == "-0" {
            "0".to_string()
        } else {
            s.to_string()
        }
    };
    format!("[{},{},{}]", n(p[0]), n(p[1]), n(p[2]))
}

/// A marker for the point-like items: doors, cube droppers, buttons, pedestals and gel droppers.
fn item_json_marker(item: &crate::level::PlacedItem) -> Option<String> {
    let (kind, tag) = match &item.kind {
        ItemKind::EntryDoor => ("door", "entry".to_string()),
        ItemKind::ExitDoor => ("door", "exit".to_string()),
        ItemKind::CubeDropper | ItemKind::IdleDropper => ("cubedropper", String::new()),
        ItemKind::FloorButton => (
            "button",
            match item.button_kind() {
                ButtonKind::Weighted => "weighted",
                ButtonKind::Cube => "cube",
                ButtonKind::Sphere => "sphere",
            }
            .to_string(),
        ),
        ItemKind::PedestalButton => (
            "pedestal",
            item.timer()
                .map_or("latch".to_string(), |t| format!("{t:.0}s")),
        ),
        ItemKind::PaintDropper => ("paintdropper", paint_name(item.paint_kind()).to_string()),
        _ => return None,
    };
    let d = item
        .normal
        .map_or(String::new(), |n| format!(",\"d\":\"{}\"", dir_name(n)));
    Some(format!(
        "{{\"k\":\"{kind}\",\"p\":{},\"t\":{},\"i\":{}{d}}}",
        cell_json(item.cell),
        json_str(&tag),
        item.index
    ))
}

fn paint_name(k: PaintKind) -> &'static str {
    match k {
        PaintKind::Bounce => "bounce",
        PaintKind::Speed => "speed",
        PaintKind::Portal => "portal",
        PaintKind::Water => "water",
    }
}

fn faces_json(faces: &[(Cell, Dir)]) -> String {
    faces
        .iter()
        .map(|&(c, d)| format!("[{},\"{}\"]", cell_json(c), dir_name(d)))
        .collect::<Vec<_>>()
        .join(",")
}

fn cell_json(c: Cell) -> String {
    format!("[{},{},{}]", c[0], c[1], c[2])
}

fn dir_name(d: Dir) -> &'static str {
    match d {
        Dir::NegX => "-x",
        Dir::PosX => "+x",
        Dir::NegY => "-y",
        Dir::PosY => "+y",
        Dir::NegZ => "-z",
        Dir::PosZ => "+z",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fields_round_trip() {
        let line = r#"{"title":"a \"quoted\" name","size":[7,5,3],"entry":{"cell":[0,0,0],"normal":"-y"},"legs":4}"#;
        let (title, size, legs) = header_fields(line).unwrap();
        assert_eq!(title, r#"a "quoted" name"#);
        assert_eq!(size, [7, 5, 3]);
        assert_eq!(legs, 4);
    }

    #[test]
    fn percent_decoding() {
        assert_eq!(
            percent_decode("%5Cout%5Cx.route.jsonl"),
            r"\out\x.route.jsonl"
        );
        assert_eq!(percent_decode("a+b"), "a b");
    }

    #[test]
    fn app_has_exactly_one_data_placeholder() {
        // Both the app page and the standalone page replace *every* occurrence with the data (or
        // `null`), so the placeholder must appear in exactly one place: the data's.
        assert_eq!(APP.matches(PLACEHOLDER).count(), 1);
        assert!(APP.contains(&format!("const DATA = {PLACEHOLDER};")));
        assert!(!APP.replace(PLACEHOLDER, "null").contains(PLACEHOLDER));
    }

    #[test]
    fn geometry_has_the_fields_the_app_reads() {
        let puzzle = crate::fixtures::bounce_gel();
        let level = Level::from_puzzle(&puzzle);
        let json = geometry_json(&level);
        assert!(json.contains(&format!(
            "\"size\":[{},{},{}]",
            level.size[0], level.size[1], level.size[2]
        )));
        for field in [
            "solid", "goo", "white", "items", "barriers", "hazards", "paint", "links", "beams",
        ] {
            assert!(json.contains(&format!("\"{field}\":")), "missing {field}");
        }
        assert!(json.contains("\"k\":\"door\""), "no entry/exit marker");
    }

    #[test]
    fn item_json_carries_the_geometry_and_every_leg() {
        let puzzle = crate::fixtures::bounce_gel();
        let level = Level::from_puzzle(&puzzle);
        let r = crate::route::route(&level, crate::solver::DEFAULT_MAX_STATES).unwrap();
        let lines = r.json_lines(puzzle.title(), level.size);

        let dir = std::env::temp_dir().join(format!("glados-view-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p2c = dir.join("x.p2c");
        let route_file = dir.join("x.route.jsonl");
        std::fs::write(&p2c, puzzle.to_p2c()).unwrap();
        std::fs::write(&route_file, lines.join("\n")).unwrap();

        let found = Mutex::new(vec![Found {
            route: route_file.clone(),
            level: p2c.clone(),
            group: "t".into(),
            title: puzzle.title().into(),
            size: level.size,
            legs: lines.len() - 1,
        }]);
        let json = item_json(&found, &route_file.display().to_string()).unwrap();
        assert!(json.contains("\"geometry\":"));
        assert!(json.contains("\"route\":{"));
        assert!(json.contains("\"header\":"));
        // One leg object per route line after the header.
        assert_eq!(json.matches("\"leg\":").count(), lines.len() - 1);
        // A path we didn't find is an error, not a panic.
        assert!(item_json(&found, "nowhere.route.jsonl").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
