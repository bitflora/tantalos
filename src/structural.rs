//! Dependency mutations of existing chambers. Geometry and item counts stay fixed;
//! mutations reroute controls or couple two existing mechanisms.
use crate::{
    level::{ItemKind, Level},
    p2c::Puzzle,
    plan::{self, Step},
};
use std::collections::HashSet;

#[derive(Clone)]
pub struct Mutation {
    pub puzzle: Puzzle,
    pub description: String,
}

pub fn mutations(parent: &Puzzle) -> Vec<Mutation> {
    let level = Level::from_puzzle(parent);
    let senders: Vec<_> = level
        .items
        .iter()
        .filter(|i| i.kind.is_sender())
        .map(|i| i.index)
        .collect();
    let wires: Vec<_> = parent
        .connections
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if c.get("Type") != Some("CONNECTION_STANDARD") {
                return None;
            }
            let (s, r) = (c.get_i32("Sender")?, c.get_i32("Receiver")?);
            let target = level.items.iter().find(|x| x.index == r)?;
            // Adding exit prerequisites tends to breed chores. Change the machinery.
            (target.kind != ItemKind::ExitDoor).then_some((i, s, r))
        })
        .collect();
    let describe = |id| {
        level
            .items
            .iter()
            .find(|i| i.index == id)
            .map_or(format!("item {id}"), |i| {
                format!("{:?}@{:?}", i.kind, i.cell)
            })
    };
    let mut out = Vec::new();
    let existing: HashSet<_> = parent
        .connections
        .iter()
        .filter(|c| c.get("Type") == Some("CONNECTION_STANDARD"))
        .filter_map(|c| Some((c.get_i32("Sender")?, c.get_i32("Receiver")?)))
        .collect();
    for &(i, s, r) in &wires {
        for &other in &senders {
            if other == s || other == r || existing.contains(&(other, r)) {
                continue;
            }
            let mut p = parent.clone();
            p.connections[i].set("Sender", other.to_string());
            out.push(Mutation {
                puzzle: p,
                description: format!(
                    "rewire {}: {} -> {}",
                    describe(r),
                    describe(s),
                    describe(other)
                ),
            });
            let mut p = parent.clone();
            let mut extra = p.connections[i].clone();
            extra.set("Sender", other.to_string());
            p.connections.push(extra);
            let item = p.items.iter_mut().find(|it| it.index() == Some(r)).unwrap();
            item.set(
                "ITEM_PROPERTY_CONNECTION_COUNT",
                (item.get_i32("ITEM_PROPERTY_CONNECTION_COUNT").unwrap_or(0) + 1).to_string(),
            );
            out.push(Mutation {
                puzzle: p,
                description: format!(
                    "couple {}: require {} AND {}",
                    describe(r),
                    describe(s),
                    describe(other)
                ),
            });
        }
    }
    for a in 0..wires.len() {
        for b in a + 1..wires.len() {
            let (i, s, r) = wires[a];
            let (j, t, u) = wires[b];
            if s == t
                || r == u
                || s == u
                || t == r
                || existing.contains(&(t, r))
                || existing.contains(&(s, u))
            {
                continue;
            }
            let mut p = parent.clone();
            p.connections[i].set("Sender", t.to_string());
            p.connections[j].set("Sender", s.to_string());
            out.push(Mutation {
                puzzle: p,
                description: format!(
                    "exchange controls: {} -> {}; {} -> {}",
                    describe(t),
                    describe(r),
                    describe(s),
                    describe(u)
                ),
            });
        }
    }
    out
}

/// Compare resource/mechanism operations, ignoring ordinary movement and portal
/// aiming. This prevents a shifted aim or a longer walk being called new logic.
pub fn operations(steps: &[Step]) -> Vec<String> {
    steps
        .iter()
        .filter(|s| !matches!(s, Step::Walk(..) | Step::Fire(..) | Step::Enter(..)))
        .map(plan::step_text)
        .collect()
}

pub fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut next = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            next[j + 1] = (row[j + 1] + 1)
                .min(next[j] + 1)
                .min(row[j] + usize::from(x != y));
        }
        row = next;
    }
    row[b.len()]
}
