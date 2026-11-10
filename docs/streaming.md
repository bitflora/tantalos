# Streaming Tantalos on Twitch

`train/broadcast.py` runs the show. Each round goes like this:

1. The level net writes about 2,000 levels at the round's difficulty tier, and the ones that are solvable, need portals and are new are kept.
2. Goblin (the net that writes plans; the solver net before 2026-10-27) writes a solution for each kept level.
3. Portal 2's editor checks the kept levels.
4. About 30 of them are played in the game, about two hours' worth. A level the net solved is played with the net's plan. For one it didn't solve, the stream shows the net's plan as far as it goes and its failed step, then Tantalos's reference solution.
5. Both nets train on the round's levels, about 75 minutes, and the training is shown on stream.
6. The next round moves to harder levels once Goblin solves at least half of the current tier.

You run OBS and go live yourself. The loop only plays the game and serves the overlay.

## Starting

```bash
.venv/Scripts/python train/broadcast.py start live-01
```

| Command | What it does |
| --- | --- |
| `.venv/Scripts/python train/broadcast.py status live-01` | Shows what the loop is doing. |
| `.venv/Scripts/python train/broadcast.py stop live-01` | Stops after the current level, or ends the current training run. |
| `.venv/Scripts/python train/broadcast.py wait live-01` | Blocks until the loop ends. |

`stop` always ends cleanly and puts your Puzzle Maker autosaves back. **Never kill the Python process.** If you must, kill `portal2.exe` first.

Options for `start`:

| Option | Default | Effect |
| --- | --- | --- |
| `--rounds N` | until stopped | Number of rounds to run. |
| `--play-hours` | 2 | Time spent playing levels each round. |
| `--queue` | 40 | Levels lined up for each round. |
| `--no-train` | off | Only play. It still writes the training files. |
| `--tier` | 1 (T2) | Starting tier, 0–3. |
| `--gen` | `v5-fresh-10h` | The starting level net. |
| `--goblin` | `solve-fresh-3h` | The starting Goblin model (`--solve` still works). |

The loop refuses to start while a training run is going, and `run.py` refuses to start while a broadcast is going. They share one GPU.

## OBS setup (1920×1080 canvas)

1. **Game Capture** (not Window Capture):
   - Mode: *Capture specific window*
   - Window: `[portal2.exe]: PORTAL 2`
   - Window match priority: *Match title, otherwise find window of same executable*

   The loop restarts Portal 2 every 20 levels and after a crash. Game Capture picks the new window up again; Window Capture loses it.
   - Position: 0, 0. The game window is 1280×800, so leave it at that size, top left.
2. **Browser source**:
   - URL: `http://localhost:8765/`
   - Size: 1920×1080
   - Turn off *Shutdown source when not visible*.
   - Put it **above** the game capture.

   The overlay leaves the game's 1280×800 area transparent while a level plays. Between levels (designing, editor check, training) it covers that area with its own screen. Add `?solid` to the URL to check it in a normal browser.
3. **Output**:
   - NVENC (hardware) encoder, so the CPU stays free for the solver.
   - 1080p30 at about 6,000 kbps suits Twitch.
   - The game and training never run at the same time, so the GPU has room for encoding.
4. Optionally, an audio source for the game.

## While live

- **Don't use the PC.** play.py brings the Portal 2 window back to the front before every step. A level built in the background never starts. Focus stealing doesn't affect OBS's capture.
- Windows notifications and sleep: turn on *Do not disturb*. run.py keeps the PC awake while training; keep the power plan from sleeping during play too.
- What the overlay shows:
  - Side panel: the level card (preview, size, shortest solution, mechanics) and the net's plan step by step: ✓ done, ↻ needed a careful retry, ✗ failed. A failed net plan is followed by the reference solution.
  - Bottom: chambers played, the share the net solved, the share played cleanly, a chart per round (net solve rate and tier), and a log.

## Where results go

| Place | What's there |
| --- | --- |
| `out/broadcast/NAME/episodes.jsonl` | One row per level played. |
| `out/broadcast/NAME/rounds.jsonl` | One row per round. |
| `out/broadcast/NAME/model-gaps.jsonl` | Reference steps that failed even when played carefully. These may be gaps in the solver's rules; they're for you to judge, and nothing changes the rules by itself. |
| `out/broadcast/NAME/round-RR/play/` | Each level's evidence (legs, trace, screenshots). |
| `data/harvest/broadcast-NAME-RR.jsonl` | Training data for the level net. |
| `data/plans/broadcast-NAME-RR.jsonl` | Training data for Goblin. |
| `runs/bc-NAME-rRR-gen`, `runs/bc-NAME-rRR-goblin` (`-solve` before 2026-10-27) | The round's training runs, recorded in `train/runs.csv` and `train/goblin-runs.csv`. |
