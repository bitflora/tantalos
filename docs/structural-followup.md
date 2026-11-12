# Structural generation follow-up

The first scale pilot sampled 120 archived Workshop sources. It screened 5,642
edits and retained 48 variants across five source chambers after checking for
shortcuts and replaying archived solutions. Two source chambers contributed
41 variants. One of the five crashed on editor export, as did the unchanged
tokenized source and three alternate mutations. The four staged variants
represented four different sources.

| Playtest | Sampling | User verdict |
|---|---|---|
| Structural Pilot 01 | Random | Solvable, boring |
| Structural Pilot 02 | Random | Solvable, interesting |
| Structural Pilot 03 | Random | Boring, solvability unreported |
| Structural Pilot 04 | Ranked | Solvable, interesting |

The two boring variants were logic-gate demos with routes dominated by cube
pickup and putdown. One interesting variant used a funnel and a pedestal
button. The other used portals and a cube. These ratings motivated testing a
broader range of source mechanisms. Four playtests are too few to estimate a
quality rate or justify a hard filter on any one mechanic.

The follow-up retries the 54 sources that failed the previous parent check
and compares two fresh groups of 90 archived training sources each. The
control is drawn uniformly from eligible source IDs. The targeted group has
at least three control wires and is spread across combinations of mechanism
types. Both start from Workshop records with archived solutions of at least
12 macro-actions. Both use the same mutation and audit settings. This
compares source selection on the available corpus. It does not measure
training uplift.

`tantalos-structural recover PARENT.tokens.txt PARENT.plans.txt` tries the
archived solutions one per line. It requires conservative replay and route
grounding. If none pass, it tries up to eight fresh shortest solutions
with a 250,000-state search budget. The fresh route must ground and replay.
The original mutation command is unchanged.

Run files and scripts are under `out/structural-followup/`. The qualified
pool is intended for review. Each record includes its Workshop source ID,
which should be kept together with the original when constructing a split.

The follow-up processed 234 sources and screened 12,039 edits. Its final
filter requires modeled solutions in four configurations and rejects
variants that can still be solved with a recorded original plan. The
archive replay uses both conservative and liberal models.

| Source group | Tried | Reached mutation | Qualified variants | Distinct yielding sources |
|---|---:|---:|---:|---:|
| Previously rejected | 54 | 12 | 3 | 1 |
| Fresh random | 90 | 61 | 49 | 6 |
| Fresh mechanism-targeted | 90 | 53 | 22 | 3 |

Alternate archived plans recovered all 12 of the previously rejected
sources that reached mutation. Fresh search found usable routes for two
sources in the fresh groups. In this run the mechanism inventory filter
reduced yield, both before and after accounting for parent eligibility.
The groups are small and these counts measure solver-screened supply.

The largest single rejection category was an unchanged original solution
(6,229 edits). Editor placement rules rejected 3,174. Search returned no
solution or repair witness for 553. The unchanged-plan check is already
early and cheap. Using the original route to propose edits could increase
the share of proposals that alter actual decisions. Improving route
grounding can expand the set of eligible sources, but the recovery
experiment added only one yielding source.

The follow-up selection uses one variant per source and prioritizes
differences in the mechanics used by the route. The action mix is a ranking
heuristic based on the playtest feedback. Human ratings remain the measure
of whether this selection produces more interesting puzzles.


All eight selected variants passed Portal 2 editor load and export. They
were staged as Structural Followup 01–08 with source attribution. The
editor check establishes that the levels load and export. The solve
claims for this batch come from the model checks.

## Overnight generation

The batch driver is `tools/workshop/structural_batch.py`. It reads archived
Workshop training records with solutions and excludes held-out IDs, user
exclusions, and sources already used in the pilots. The primary queue starts
from archived solutions of at least 12 actions and at least two non-exit
control wires. The expanded queue tries the other sources with a non-exit
control wire. It preserves the same acceptance criteria for generated levels.

Each source runs through solution recovery, structural mutation, four solver
configurations, and conservative and liberal replay of the archived original
plans. A kept variant has a solution of at least 16 macro-actions in all four
configurations, and none of the recorded original plans still solve it in the
replay checks. The batch saves at most two variants per Workshop ID, with
distinct action-type sequences for that source.

The batch saves each finished source transactionally in `run.sqlite` and
refreshes `qualified.jsonl` and `status.json` periodically. The two queues use
separate folders under `out/`. Resume by rerunning the same command against
its folder after the previous process has ended.

```powershell
.venv/Scripts/python tools/workshop/structural_batch.py --out out/structural-overnight-2026-11-10 --hours 8 --workers 6
.venv/Scripts/python tools/workshop/structural_batch.py --out out/structural-overnight-expanded-2026-11-10 --hours 8 --workers 4 --min-parent 1 --min-wires 1 --exclude-primary
```

Exports carry the Workshop ID, parent symmetry, source URL, tokens, plan,
and solver checks. They are candidate training data. The records explicitly
mark editor and human playtest verification as false.

For the November 10 run, the combined export is
`out/structural-overnight-2026-11-10/dataset.jsonl`, with progress in
`dataset-status.json` in the same folder. The publisher deduplicates exact
level tokens across the two queues and checks the cap of two per source.
