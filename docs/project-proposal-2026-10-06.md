> Historical proposal, superseded by [the current roadmap](../plan.md).
> Estimates and implementation suggestions below describe the original feasibility study.

# Spec: Neural net that builds Portal levels (project "tantalos")

## Context
User wants feasibility spec: train a neural net to generate Portal levels. Questions: hardware, time, amateur viability. `/home/tjasko/code/tantalos` is empty — greenfield. This doc = spec + phased plan, not code yet.

## Key decision: target Portal 2 Puzzle Maker (PeTI), not raw Hammer maps
- Raw Source maps (VMF/BSP brushes, arbitrary geometry, I/O logic) = huge unstructured output space. Not amateur-viable.
- PeTI levels = **voxel grid (128-unit cells) + small fixed item vocabulary** (buttons, cubes, lasers, bridges, funnels, gels, faith plates, doors) + item connections. Stored as `.p2c` — plain-text KeyValues, VMF-like. Opens in the in-game editor, compiles to playable map for free.
- So model output = structured text/grid, small. Game engine does the rest (lighting, compile, art).

## Prior work (searched 2026-10-06)
No mature ML Portal level generator found. Closest:
- NISER student project, DCGAN on Portal 2 voxel levels (hand-built corpus + rotation augmentation): https://www.niser.ac.in/~smishra/teach/cs460/2020/project20/group23/report.html
- Analogous ML level-gen methods to copy: MarioGPT-style LLM gen (https://arxiv.org/pdf/2302.05817), TOAD-GAN (https://arxiv.org/pdf/2008.01531), DOOM GAN (https://arxiv.org/pdf/1804.09154.pdf), ScriptDoctor LLM+tree search (https://arxiv.org/pdf/2506.06524), PCGML survey (https://arxiv.org/abs/1702.00539), PCG+LLM survey (https://arxiv.org/pdf/2410.15644)
- Reusable tooling: Portal2.Puzzle C# p2c read/write (https://github.com/Kyle0654/Portal2.Puzzle), p2c_conv (https://github.com/gusarba/p2c_conv), BEE2.4 (https://github.com/BEEmod/BEE2.4), Portal2Workshop map archive, check contents (https://github.com/pansong291/Portal2Workshop), GitHub p2c topic (https://github.com/topics/p2c)
- Hand-designed Puzzle Maker test battery for fluid intelligence research, possible difficulty-labeled eval set: https://www.researchgate.net/publication/297744169

## Hard problems (ordered by pain)
1. **Data.** Workshop has a very large PeTI corpus, but published items ship compiled `.bsp`, not `.p2c` source. Prior student project (DCGAN on Portal 2) ended up hand-building a small corpus because of this. Options:
   - a. Decompile workshop BSPs (BSPSource) → reconstruct voxel grid + items. PeTI geometry is grid-aligned and items come from known instances, so reconstruction is plausible but unproven. **Biggest risk; spike first.**
   - b. Collect `.p2c` files authors share directly (forums, BEEmod community, own maps). Small (hundreds–low thousands).
   - c. Synthetic: procedural generator of simple valid chambers → pretrain, fine-tune on real.
   - Download tooling: SteamCMD `workshop_download_item 620 <id>` (Portal 2 app 620) + Steam Web API for metadata/votes. Votes = free quality label (train on top-rated only).
   - Legal: workshop maps are authors' content. Fine for private research; don't redistribute dataset or model trained on it without thought.
2. **Solvability.** Model will emit levels that *look* right but are unsolvable or trivially solvable. Need validator:
   - Structural checks (cheap): entrance/exit exist, items placed on valid surfaces, connections type-check, exit reachable at all.
   - Abstract solver (hard): search over state (player region, portal placements on portalable surfaces, button/cube states). Start with restricted item set — portals, cubes, buttons, doors, lasers. Add bridges/funnels/gels/faith plates later (physics-heavy, much harder).
   - Human playtest for final judgment. "Fun" has no metric; ratings classifier is a weak proxy.
3. **Representation.** Serialize level to compact token sequence (e.g. run-length voxel layers + item list with coords + connection list). Must round-trip losslessly back to `.p2c`.

## Recommended model approach
**Fine-tune a small open-weight LLM on serialized levels** (MarioGPT / "Level Generation Through LLMs" style).
- Base: 1–8B param open model. LoRA/QLoRA.
- Text format lets you condition on prompts later ("2 cubes, laser, medium difficulty").
- Generate → validator filter → keep survivors. Optional later: rejection-sampling fine-tune on validator-passing outputs (cheap self-improvement loop).
- Zero-training baseline first: prompt a frontier LLM API with format spec + few examples. Sets bar; may be good enough for fun.
- Skip 3D GAN/diffusion: data-hungry, no connection/logic modeling, prior attempts weak.

## Alternative: self-play (AlphaGo-style generator + solver)
Generator builds level → solver attempts → generator rewarded for "solvable but hard". No scraped data needed.

**Core requirement: fast abstract simulator.** AlphaGo works because Go rules are perfect and microsecond-fast. Portal 2 engine is slow, physics-based, not headless → unusable in loop (millions of episodes). Must write own discrete Portal model: grid cells, portalable surfaces, line-of-sight portal placement, cube/button/door state, pits. Simulator = most of project. Self-play trades the *data* problem for the *simulator* problem.

**Design (recommended if going this route):**
- Solver: **classical search** (BFS/A* over abstract state), not neural. Small chambers → exact, cheap, can't be fooled. Gives solution length, branching, # solutions = difficulty signals.
- Generator: MCTS over edit actions (place block/item/connection), AlphaZero-style policy+value net learns to predict reward and prune search. Or PCGRL-style RL agent.
- Reward: solvable (gate) × difficulty (min solution length, dead ends, "aha" steps like required portal momentum) − cheese (shortcut solutions, unused items).
- Solvability by construction trick: generate via reverse play (Kartal Sokoban MCTS) — always solvable.

**Risks:**
- Reward hacking: generator exploits gaps between abstraction and real game → "solvable" in sim, broken in Portal 2 (or vice versa). Validate samples in-game regularly.
- Hard ≠ fun. Optimizing difficulty yields mazy, ugly, unnatural chambers. Fix: hybrid — add style prior (small model trained on real levels, KL penalty / discriminator reward). Small real corpus enough for this.
- Physics items (momentum flings, funnels, bridges, gels) hard to model discretely. Start: portals + cubes + buttons + doors + pits + lasers. Momentum flings next (key to Portal feel).

**Compute:** CPU-bound (simulator + search). 16+ core CPU + one GPU for policy/value net. Abstract sim in Rust/C++ (or numba) not pure Python. Still amateur-affordable; nowhere near AlphaGo scale (search space tiny vs Go).

**Time:** simulator 2–3 mo, solver 1 mo, generator + reward tuning 2–3 mo → ~5–8 mo evenings/weekends. More engineering than LLM route, but no data risk and solvability is guaranteed by construction.

**Prior art:** Kartal et al. Sokoban MCTS gen (https://conservancy.umn.edu/items/a38495a0-b17d-4337-be34-d7855b81a8bc, data-driven follow-up https://ojs.aaai.org/index.php/AIIDE/article/view/12859), PCGRL (https://arxiv.org/pdf/2001.09212), PAIRED (https://alphaxiv.org/abs/2012.02096, blog https://blog.research.google/2021/03/paired-new-multi-agent-approach-for.html), ScriptDoctor (https://arxiv.org/pdf/2506.06524).

**Verdict:** Stronger long-term approach than pure LLM imitation, since solvability is the hard part and self-play attacks it directly. Best version = hybrid: self-play for solvability/difficulty, small real-level corpus for style.

## Solver design
Goal: given a level, answer *solvable?*, *how?*, and *how interesting?* in ≤1 s, fully automatically, millions of times. Classical search over an abstract model, no neural net. Rust core + PyO3 bindings so the ML loop calls it from Python.

### Core idea: free movement collapses, discrete events are the search
Walking around a room is continuous and boring. What matters is the discrete events: placing a portal, picking up or dropping a cube, pressing a button, going through a portal. So:
1. **Geometry precompute (per level, per geometry config):** voxel grid → walkable cells → **regions** = sets of cells mutually reachable on foot.
2. **Search over macro-states:** player position = region id only. Every action is a discrete event available from the current region.
Same trick as Sokoban solvers (player position normalized to its reachable area). It turns a continuous 3D game into a small graph search.

### Abstract model
**Static geometry** (from `.p2c`): solid/empty voxels, each face portalable (white) or not (black), goo/pits, entrance, exit.

**Physics constants** (measure once in-game with SAR HUD, don't guess): `sv_gravity` (600 default), walk speed, jump and crouch-jump height, cube height, whether cube + jump clears one 128 voxel. Long-fall boots → no fall damage; only goo kills.

**Movement edges between regions:**
| Edge | Condition |
|---|---|
| walk | adjacent cells, same height, nothing blocking (closed door, fizzler while holding a cube, active laser if conservative) |
| drop | step off a ledge, any height (lands where gravity says) |
| climb | ledge ≤ jump height; or ≤ cube+jump height if a cube is in the region |
| portal | enter portal A from a region that touches it → exit at B's landing region |
| fling | fall from height h into a floor portal → exit speed √(2gh) (capped at max velocity) → ballistic arc from B → landing region(s) |
| faith plate | fixed arc → landing region |

**Portal targeting:** a player in region R can place a portal on face F if F is portalable, at least one cell of R has line of sight from eye height to F (3D DDA raycast through voxels), and F is big enough (in PeTI a white face fits a portal). Precompute `visible[R] = {faces}`.

**Face equivalence classes (key to state size):** a chamber may have hundreds of white faces, but most behave identically. Group faces by (orientation, which regions can enter a portal there, landing region on exit, fling geometry). A typical chamber shrinks from 300+ faces to ~10–40 classes. Search over classes, not faces. Without this the state space blows up (faces² × regions × items).

**Item state:**
- Cubes: location = region, or `held`, or `on_button(X)`. Identical cubes are sorted, so permutations collapse. Cubes travel with the player through portals, can be dropped into floor portals (cube transport without the player), and are fizzled by grills back to the dropper.
- Floor buttons: pressed iff a cube sits on them, or the player stands on them (only true while the player is in that region).
- Pedestal buttons with timer: the timed door opens only if the path from button to door (shortest walking time) fits in the timer. Macro-action "press + run".
- Logic: PeTI connections (multiple inputs = AND, inverted flags) are evaluated as a pure function of item states after each action, giving door/laser/bridge on-off.
- Lasers: beam traced through voxels, redirected by reflection cubes (orientation discretized to 4 directions) and through portals; hits catcher → active.

**Dynamic geometry:** doors, light bridges and funnels change the region graph. Bridges and funnels pass through portals, so geometry depends on portal placement. Fix: geometry config = (open doors, active bridges + their portal routing). Memoize the region graph per config, computed lazily. Most levels have only a few configs.

### Search
- **State:** `(player_region, held, blue_class, orange_class, cube_locs, laser_cube_orients, toggle_states)`. Toggles = only the latched items (pedestal toggles); everything else is derived.
- **Actions:** place blue/orange (any visible class), pick up or drop a cube in the current region, put a cube on a button in the region, press a pedestal, take a movement edge (walk/drop/climb/portal/fling/plate).
- **Goal:** player in exit region with exit door open.
- **Algorithm:** BFS for minimum macro-action count (the difficulty metric). Dijkstra with walk-time costs when timers matter. A* later if needed; PeTI levels usually have ≤3 cubes, so plain BFS + hash-set dedupe should suffice.
- **Budget:** cap states (e.g. 10⁶). Hitting the cap = "unknown", not "unsolvable".
- **Alternative for quick prototyping:** export the abstract model as PDDL → Fast Downward. Free heuristics, but timers, flings and metric extraction are awkward. Custom search is preferred; geometry precompute is the real work either way.

### Soundness: two modes, three verdicts
The abstraction is wrong in both directions (tight flings, air control, funnel magnetism, speedrun tricks). So run twice:
- **Conservative** (only well-calibrated moves, wide margins on flings and jumps, lasers block): solvable here → **confident solvable**.
- **Liberal** (any plausible move, laser crossing allowed, landing tolerance widened): unsolvable even here → **confident unsolvable**.
- Mixed → **uncertain**: send to in-engine check, or discard during training.

### Outputs (what the generator is rewarded on)
| Metric | How | Why |
|---|---|---|
| solvable | verdict above | gate |
| min solution | BFS depth + action list | base difficulty |
| portal placements in solution | count | "thinking with portals" vs walking |
| item necessity | ablation: remove each item, re-solve; still solvable → unnecessary | elegance; cheese/red-herring detector |
| solution count | number of distinct plans within min+k | 1 = tight puzzle; many = loose |
| dead ends | backward reachability: % of reachable states that can't reach goal | softlock risk (PeTI droppers usually prevent these) |
| non-obviousness | success rate of a naive greedy/random player in the same model | "aha" proxy, near-free compared with an LLM |
| backtracking | solution moves away from exit (distance-to-exit rises) | hallmark of good puzzles |
| fling/timing reliance | uses edges near tolerance limits | flag for in-engine verify |

### Closing the loop: solver plan → TAS → in-game check
The solver's action list uses the same high-level API as the LLM player (`walk_to`, `fire_portal`, `pickup`...). Feed it to the SAR harness → executes in real Portal 2 → did the player reach the exit? This gives automatic sim-vs-engine validation with no LLM and no human. It also catches reward hacking (levels that are only solvable in the sim). Run on a sample of every generator batch.

### Item tiers (build order)
| Tier | Items | Notes |
|---|---|---|
| 0 | walls, white/black, goo, entrance/exit, cubes, droppers, floor buttons, doors, logic | MVP; most classic Portal 1 puzzles |
| 1 | pedestal buttons + timers, grills/fizzlers, lasers, catchers, relays, reflection cubes, faith plates | beam tracing, timing |
| 2 | flings, light bridges, turrets | ballistics; dynamic geometry through portals |
| 3 | funnels, gels, moving platforms | gels = per-face paint state explosion; heuristics only |

### Worked example
Room A (entrance, floor) and a high ledge B (exit). White wall in A, white wall on B visible from A.
- Precompute: regions {A, B}; A→B unreachable on foot (too high); classes `a` (wall in A), `b` (wall on B), both visible from A.
- BFS: `(A,-,-,-)` → place blue `a` → place orange `b` → portal edge A→B → goal. Min solution = 3.
- Add: exit door wired to a button on B, cube dropper in A. The solver must pick up the cube, carry it through the portal, and drop it on the button. Min 6. Ablation shows the cube is necessary. Good.

### Verification
- Hand-encode 10–20 known chambers (Portal 1 test chambers, PeTI classics). All must be solvable, with solution lengths matching intent.
- Mutation tests: remove a required item or paint the key wall black → must flip to unsolvable.
- Fuzz: random levels → solver plan → in-engine replay. Track agreement rate per item tier. That rate is how much to trust the solver.
- Perf: <1 s median per level on one core → ~1M levels/day on 16 cores.

### Effort
Tier 0 geometry + search: 3–5 wk. Face classes + soundness modes: 2 wk. Tier 1: 3–4 wk. Tier 2 (flings, bridges): 4–6 wk, research-y. Matches phase 5 estimate; tiers 2–3 push past it.

## LLM playtester: generator builds, Claude plays (researched 2026-10-06)
Idea: a frontier model (Opus 5.5) plays each generated level. It complements the BFS solver and doesn't replace it.
- **BFS solver** = proof. Is the level solvable, what is the shortest solution, what shortcuts exist. Cheap, exact, inner loop.
- **LLM player** = discoverability proxy. Can a reasoning agent *figure it out* from what it sees? Gives attempts, wrong hypotheses, time-to-"aha". Closer to "fair + interesting" than solution length is.
- An LLM failure ≠ bad level. It can be a perception failure, an action-API limit, or bad luck. Never use it as the only solvability judge.

### Two play modes
| Mode | Observation | Actions | Use |
|---|---|---|---|
| **Abstract** (text) | serialized grid + item states from own simulator | `move_to(cell)`, `portal(color, surface_id)`, `pickup`, `drop`, `press` | training signal, bulk, parallel, cheap, no game needed |
| **In-engine** (Portal 2) | screenshot + structured overlay | same high-level API compiled to real inputs | the show; catches sim-vs-game gaps (reward hacking check) |

### Learning loop with an LLM in it
- Generator → structural validator → BFS gate → LLM player → score → generator update.
- **PAIRED-style reward** (prior art already cited): use two players, strong (Opus 5.5) + weak (Haiku 4.5 or Sonnet 5.5). Reward = strong solves AND weak fails. That pushes toward levels that need reasoning, not obscurity or mazes. Unsolvable-by-both = penalty even if BFS-solvable (likely unfair or unreadable).
- Generator update by type:
  - Fine-tuned small model → rejection-sampling FT or DPO pairs (preferred = solved-with-struggle, rejected = trivial/unsolvable).
  - Opus-as-generator (zero training) → no weight updates. "Learning" = a persistent design notebook of what worked or failed, fed back in context. Weaker, but works on day 1 and makes a good stream story.
- Cost keeps LLM out of the inner loop. BFS scores millions of levels; LLM plays only the top few hundred per generation.
- Risk: generator learns LLM blind spots (visual confusion, API quirks) instead of real puzzle depth. Mitigate: weak/strong *difference* reward, periodic human playtests, vision-only + full-info runs compared.

### In-engine harness (hardest part of the show)
Portal is real-time 3D first-person. LLM steps take seconds. Raw mouse/keyboard play is not viable. Copy the Claude Plays Pokémon pattern: few tools, high-level navigation, screenshot + parsed state overlay, knowledge-base memory + summarization.
- **Pause-step loop:** game paused → screenshot + state → model picks action → harness executes it → pause again.
- **Execution via SourceAutoRecord (SAR) TAS:** SAR plays `.p2tas` scripts (`sar_tas_play`), with tick-advance/pause and auto-aim/auto-strafe tools, Windows + Linux. Harness compiles each high-level action into framebulks. Path planning uses own `.p2c` grid, so geometry is known exactly. No `setpos` teleports, so moves stay legit and replayable.
- **Tools exposed to model:** `walk_to(target)`, `look_at(target)`, `fire_portal(color, surface_id)`, `use`, `jump`, `crouch`, `wait(s)`, `update_notes`, `give_up`. Flings, funnels and timing puzzles come later, maybe as `fling(in_portal, out_portal)` macros.
- **Observation:** screenshot (~1280×720) + text overlay: player pos/velocity, held object, both portal positions, button/door/laser states, list of visible portalable surfaces with IDs. Source: `.p2c` (static) + SAR HUD / console queries (dynamic).
- **Honesty knob:** "full-info" (overlay lists everything) vs "vision-only" (screenshot + pos). Run both; the gap between them is interesting data and content.
- **Level load pipeline:** puzzle maker rebuild exports `Preview.vmf` → vbsp/vvis/vrad → `Preview.bsp` in `maps/puzzlemaker`; `changelevel` loads it. **Spike:** script `.p2c` → bsp without clicking through the editor GUI (call compilers directly on our own vmf export, or BEE2's compiler chain). `vrad -final` is slow, so precompile the next level while the current one plays.
- **Control channel:** console commands + SAR TAS server socket (p2tas VS Code plugin already talks to it, so the protocol exists). Verify whether `-netconport` remote console works in Portal 2.
- **Replays:** record a demo per attempt. Free highlight reels; failures are debuggable.
- Platform: see "Platform" section.

### Cost per playthrough (API list prices cached 2026-09-25)
Opus 5.5 $4/$20 per MTok in/out, cache read $0.20. Sonnet 5.5 $2/$10. Haiku 4.5 $1/$5. Screenshot ≈ (w×h)/750 tokens → 1280×720 ≈ 1.2k.
| | Per action | Per level (40–150 actions) | Per stream hour (~1 action/10s) |
|---|---|---|---|
| Opus 5.5, cached history | ~$0.03–0.08 | ~$1.50–10 | ~$10–30 |
| Sonnet 5.5 | ~half of Opus | ~$0.75–5 | ~$5–15 |
| Abstract text mode, Opus | ~$0.01–0.02 | ~$0.20–1 | n/a |
Rough estimates, assuming ~30k cached context + ~2k new input + ~1–3k output per step. Measure on 10 real runs before budgeting. Prompt caching is mandatory (stable system prompt + tool list first). Effort default on Opus 5.5 is `medium`; set it explicitly.
- 24/7 Opus stream ≈ $250–700/day. Unaffordable for a hobby. Do scheduled blocks (e.g. 3 h/night ≈ $30–90).
- Cheaper split: Sonnet 5.5 executes moves, Opus 5.5 plans (advisor tool / planner call only on new room or when stuck).

### Broadcasting on Twitch
**Pacing is the main problem.** LLM thinking = dead air while the game is paused. Options, best first:
1. **Solve-then-replay with captions.** Agent solves offline (paused stepping). Stream plays the recorded demo in real time with the model's reasoning synced as captions. No dead air, no live failures, cheap, retimeable. Costs liveness.
2. **Live with thinking panel.** Paused game + streamed reasoning text (`thinking.display: "summarized"` on the API, or the model's own narration between tool calls). Honest and live, slow.
3. **Hybrid.** Live "designer builds level" segment (generator reasoning + level appearing in editor view), then replay of the solve attempt, then live verdict + score.

**Stack:**
- OBS (obs-websocket built in) on the game PC. Harness switches scenes (designing / compiling / playing / verdict).
- Overlays = local web app as OBS browser source, fed by harness over websocket: reasoning feed, top-down minimap from `.p2c` grid with player + portals, level card (generator prompt, BFS min-solution length, attempt #, cost so far), scoreboard (solved / gave up / unfair, generator win-rate vs player).
- Twitch EventSub for chat + channel-point Predictions API: "Will Opus solve it?" bets, viewer votes on next-level constraints ("2 cubes + laser, hard").
- **Never pipe raw chat into the model prompt.** Prompt injection, plus the model might repeat abusive chat on stream. Viewers choose only from fixed vote options.
- Optional TTS narrator with an original voice. Don't clone Tantalos / Ellen McLain's voice (impersonation + rights).

**Reliability for unattended runs:** watchdog restarts game on crash. Stuck detection: no state change in N actions → `give_up` → mark level → next. Per-level and per-day spend caps. API error/refusal fallback (server-side `fallbacks`). Pre-queue 3+ compiled levels.

**Policy / legal:**
- Valve video policy: non-commercial use OK, and monetizing via platform partner programs is explicitly allowed. Don't paywall the stream (sub-only = gray area).
- Twitch has no AI-disclosure rule yet. Label it as AI anyway ("AI-generated levels, AI player").
- Unattended streams are fine if moderation stays on (AutoMod, mods, any chat-driven TTS disabled when away).
- Twitch opted channels into Amazon gen-AI training by default (Aug 2026). The opt-out is in settings; your call.
- Tantalos/Aperture = Valve IP. Fan framing is common, but don't imply Valve affiliation; pick a channel name that isn't just "Tantalos".
- Workshop-derived training data: don't stream others' workshop levels as your own; generated levels are fine.

**Hardware:** gaming PC runs Portal 2 + OBS (NVENC encode) easily. ~6 Mbps upload for 1080p60. Harness is light (API calls). LLM cost >> hardware cost.

## TODO: raise Workshop yield

Support these, in rough order of how many top-rated puzzles they block (1,798-puzzle crawl, 2026-10-12):
- **Lasers**: emitters, catchers, relays (`laseremit`, `lasercatch`, `laserrelay`), and reflection cubes redirecting them.
- **Panels**: angled panels (`angledPanel`, `angledClearPanel`) and flip panels (`flipPanel`).
- **Buttons wired to cube droppers**: currently rejected as "wired to cubedropperN". Needs a
  `> dropper N` target in the tokens and a solver rule for re-dropping.
- **Wall-mounted gel droppers**: currently rejected as "gel dropper not on a ceiling". Gel
  sprayed sideways, not just poured onto the floor below.

Each one needs a token in src/tokens.rs, decoding in src/level.rs, solver support, and a matching rule in tools/workshop/convert.py.

## TODO: check the solver against Workshop maps it can't solve

The 2026-10-14 bulk build (`out/workshop/levels.jsonl`, 5,378 converted) gave 837 solvable, 1,610 unsolvable, 2,108 uncertain and 789 invalid. Published maps were completed by their authors, so unsolvable and uncertain mean a solver or converter gap, not a bad puzzle.
- **Uncertain (2,108)**: probably the search budget. Rerun `fetch.py build --budget <larger>` on them, then look at which stay uncertain.
- **Unsolvable (1,610)**: sample some, play the author's solution in game (`train/play.py`), and find which item or mechanic the solver lacks. Start with the assumptions in README "Unverified" (panel hinge, bridge height, funnels, rails).
- **Invalid (789)**: check whether the converter or the editor-rules model is wrong.
- Fixes in `src/`, or in `tools/workshop/convert.py` for conversion bugs, then rebuild. Each recovered level adds training data (`data/ws2`).

### Phases (add after phase 5)
| Phase | Work | Est. |
|---|---|---|
| 5a Abstract player | Opus/Haiku play abstract sim via tool API; compare with BFS on hand-made chambers | 1–2 wk |
| 5b Harness spike | Scripted `.p2c` → bsp; SAR TAS executes `walk_to` / `fire_portal` reliably on one PeTI map | 2–4 wk (**riskiest**) |
| 5c In-engine player | Full tool loop, overlay, demos, stuck detection; 10 levels end-to-end | 2–4 wk |
| 7 Stream MVP | OBS scenes + browser overlay + solve-then-replay format; private test streams | 2–3 wk |
| 8 Interactive | Chat votes, Predictions, live segments, PAIRED scoreboard | ongoing |
Order: abstract player first. It's useful for training even if the stream never happens, and it de-risks the action API before fighting the engine.

**Prior art:** Claude Plays Pokémon harness (3 tools: button/knowledge-base/navigator, RAM overlay, summarization; ~35k actions for 3 badges): https://michaelyliu6.github.io/posts/claude-plays-pokemon/, https://www.zenml.io/llmops-database/building-and-deploying-a-pokemon-playing-llm-agent-at-anthropic. Game-agent benchmarks note brittle vision, and some models do *worse* with images than with text state: BALROG https://arxiv.org/pdf/2411.13543, lmgame-Bench https://arxiv.org/html/2505.15146v2. SAR + p2tas: https://sar.portal2.sr/, https://wiki.portal2.sr/TAS, https://github.com/RainbowwPhoenixx/p2tas-lang. PeTI compile chain: https://developer.valvesoftware.com/wiki/P2C. No prior LLM-plays-Portal project found.

## Platform (recommendation 2026-10-08, not yet decided: native Windows 11)
Only the game-touching parts care about OS: compiling `.p2c` → bsp, SAR harness, OBS/stream. The solver, generator, serializer and training are pure code and run anywhere.
| | Windows 11 native | Linux native | WSL2 (current) | macOS |
|---|---|---|---|---|
| Portal 2 | native, primary | native (OpenGL/Vulkan) or Proton | runs on host only | **dead**: 32-bit build, won't run since Catalina |
| Puzzle maker compile (vbsp/vvis/vrad) | native, what everyone uses | `vbsp_linux` exists, known crashes (shaderapiempty) | host `.exe`, called across boundary | n/a |
| BEE2 / Hammer / BSPSource | all work | BEE2 from source, Hammer via Wine, BSPSource (Java) ok | host | n/a |
| SAR + TAS | yes | yes | host | n/a |
| OBS + NVENC | best game capture | good (PipeWire) | host | n/a |
| Rust / Python / Claude Code | all native | best | good | — |
| GPU training (CUDA) | ok (some libs Linux-first) | best | good | — |
| Friction | none for game; slightly clunkier shell | compile-pipeline risk | two OSes, cross-boundary paths/sockets, GPU shared | — |

**Recommendation: everything on native Windows 11** (game, harness, OBS, Rust/Python dev, Claude Code). Harness ↔ game on one OS = no cross-boundary sockets or path translation. The Portal 2 modding ecosystem is Windows-first, and the riskiest spike (scripted compile + SAR control) is in that ecosystem.
- Training: rent a Linux cloud GPU per run (~$20–200), or keep WSL2 just for CUDA training. Don't block on local ML setup.
- Linux-native alternative, if preferred: viable *only if* day-1 test passes (build a PeTI level, compile with `vbsp_linux`/vvis/vrad, load, run SAR TAS). Fallback: Windows Portal 2 under Proton with the `.exe` compilers under Wine. More moving parts.
- Avoid split setups (game on one OS, harness in another). This is the current WSL arrangement and costs friction every day.
- Screenshots via OBS websocket `GetSourceScreenshot` keep the harness OS-agnostic either way.

## Hardware
| Setup | Use | Cost |
|---|---|---|
| Any modern PC + Portal 2 | data pipeline, validator, playtest | have it |
| 1× 24 GB GPU (RTX 3090 used / 4090 / 5090) | QLoRA 1–8B model, inference | ~$800–2500 one-time |
| Or rent: 1× A100/H100 cloud | same, faster | ~$1–3/hr; ~$20–200 per training run |
| 16 GB GPU | QLoRA ≤3B only | ok for prototyping |
Not needed: multi-GPU cluster, pretraining from scratch. Disk: ~100s GB if mass-downloading BSPs.

Training scale estimate: 10k–50k levels × 2–8k tokens ≈ 50–400M tokens. 1–3B model on one 4090: hours to ~1 day per epoch. 8B QLoRA: ~1–3 days. Small data (hand corpus, ~1k levels): under an hour.

## Time (amateur, evenings/weekends)
| Phase | Work | Est. |
|---|---|---|
| 0 Spike | Parse `.p2c`; test BSP→grid reconstruction on 20 workshop maps | 1–2 wk |
| 1 Format | Serializer + `.p2c` writer, round-trip test, opens in editor | 1–2 wk |
| 2 Data | Mass download + reconstruct + filter by votes/item set | 3–8 wk (if 0a works) |
| 3 Baseline | Frontier-LLM prompt baseline; structural validator | 1–2 wk |
| 4 Train | QLoRA fine-tune, sample, eval | 1–3 wk |
| 5 Solver | Abstract solver, restricted item set | 4–12 wk (research-y) |
| 6 Loop | Filter + rejection-sampling fine-tune; conditioning prompts | ongoing |
**Total:** ~2–4 months → model emits editable chambers that load in editor, mostly structurally valid. Reliably solvable + fun = 6+ months, open research.

## Viability verdict
- **Viable for amateur** with Python + basic ML skill, *if* scoped to PeTI + restricted items.
- Compute is not the bottleneck (one consumer GPU or ~$100s cloud).
- Bottlenecks: (1) getting data out of workshop BSPs, (2) solvability checking. Both are engineering/research, not GPU problems.
- Not viable: full Hammer-quality Portal 1/2 maps with custom geometry.

## Proposed repo layout (when building)
- `tantalos/p2c/` — `.p2c` parser/writer
- `tantalos/serialize/` — level ↔ token format
- `tantalos/data/` — SteamCMD fetch, BSP decompile + grid reconstruction, vote filter
- `tantalos/validate/` — structural checks, abstract solver
- `tantalos/train/` — QLoRA fine-tune + sampling scripts
- `tantalos/eval/` — metrics, playtest export
- `tantalos/play/` — LLM player: tool API, abstract-sim adapter, SAR/TAS engine adapter, overlay state, memory
- `tantalos/stream/` — OBS websocket control, browser overlay app, Twitch EventSub/Predictions, replay captioning

## Verification
- Round-trip: every `.p2c` → tokens → `.p2c` is identical; written files open in Portal 2 editor and compile.
- Reconstruction accuracy: on maps where both `.p2c` and `.bsp` exist (your own), compare reconstructed grid/items to source.
- Model metrics: % samples parse, % pass structural validator, % pass solver, novelty (distance to nearest training level — catch memorization).
- Human: playtest 20 sampled levels per checkpoint; score solvable / interesting.

## Recommended first step
Phase 0 spike: download ~20 workshop PeTI maps, run BSPSource, check whether voxel grid + items recoverable. Result decides data strategy (large corpus vs small hand + synthetic).
If self-play route: first step = abstract simulator + BFS solver for minimal item set (portals, cube, button, door, pit), hand-encode 5 Portal 1 chambers, confirm solver finds the intended solutions. Plus `.p2c` writer so sim levels can be playtested in-game.
