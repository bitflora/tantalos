# Overnight job: one full Goblin self-play round (E1 round 0)

For an agent running this without earlier context. Follow these steps in order. Don't improvise:
if something here doesn't match what you see, stop and tell the user.

## What it is

`train/school.py` makes Goblin (the plan-writing net) try levels our Rust solver can't finish.
The plans the checker accepts go into a buffer, to train on later. This is one full round:

- 4,096 levels;
- for each level, 1 greedy plan and 32 sampled ones;
- then prefix resampling, the liberal check and polish.

It is expected to take **7-9 hours**. The 256-level rounds took about 20 minutes each; see
`docs/goblin-school-next-2026-10-29.md`.

It saves after every chunk of levels. If it is stopped, the same command resumes it, and at
most one chunk is lost.

## Rules

- Don't train, don't run `cargo build`, and don't start any other GPU or heavy job while it
  runs. It uses the GPU and `target/release/glados-score.exe`.
- Don't change `train/school.py` or any option of the command below.
- Don't delete anything under `runs/school/`.
- Don't commit or push. The round's outputs under `runs/` are not committed.
- Don't kill processes you didn't start. If the round has to be stopped and you didn't start it,
  ask the user.

## 1. Before starting

Run each of these. Stop and tell the user if any check fails.

1. Nothing else is using the GPU:

   ```bash
   nvidia-smi
   ```

   The GPU should be using under about 1.5 GB and near 0%. If it isn't, ask the user what is
   running. A training run, a broadcast or another school round all count.
2. The branch is `goblin-env` and the tree is clean:

   ```bash
   git status --short --branch
   ```

3. The model and the binary exist:

   ```bash
   ls runs/goblin-v6init-12h-b/best.pt target/release/glados-score.exe
   ```

4. The round hasn't been started before:

   ```bash
   ls runs/school/e1
   ```

   - If the folder doesn't exist, go on.
   - If `runs/school/e1/r00/summary.json` exists, the round is already finished: skip to step 3.
   - If `runs/school/e1/r00/` exists without `summary.json`, the round was interrupted. Use the
     command in step 2 unchanged: it resumes.

5. The tests pass. Each command must end in `OK` or `test result: ok` with no failures:

   ```bash
   .venv/Scripts/python -m unittest discover -s tests -p test_school.py
   ```

   ```bash
   cargo test --release --lib env::
   ```

   The second command may compile the tests. That is fine before the round, never during it.

## 2. Start the round

Run this exactly, with the Bash tool's `run_in_background: true`:

```bash
.venv/Scripts/python train/school.py round e1 --model runs/goblin-v6init-12h-b/best.pt --round 0 > runs/school-e1-r00.log 2>&1
```

The default `--levels` (4096) and `--k` (32) are the ones wanted; add nothing.

Tell the user it has started. You are notified when it ends. Don't poll it and don't sleep. If
the user asks for progress, look once:

```bash
tail -5 runs/school-e1-r00.log
```

It prints:

- `attempt: N of 4096 levels` every 128 levels;
- `prefix pass 1: N levels` and `prefix pass 2: N levels`;
- finally a summary table and `wrote ...runs\school\e1\r00`.

**If the user wants the machine back:** stop the background task (TaskStop on its id). Tell the
user it is saved and resumable with the same command. Don't delete anything.

**If it fails** (the notification shows a non-zero exit code, or the log ends in a Python
traceback):

1. Copy the last 30 lines of `runs/school-e1-r00.log` to the user.
2. If the error is `glados-score env stopped` or `memory allocation ... failed`, run the same
   command again once: it resumes.
3. If it fails a second time, or the error is anything else, stop and report. Don't edit code to
   work around it.

**If it exits at once** with `was started with --seed ...` or `was started with --model ... --k
...`, you changed the command: use exactly the one above. With `exists but has no
progress.json`, or `saved round contains held-out test/probe identities`, stop and tell the
user; don't delete the folder.

## 3. When it finishes: check the results

1. Read the summary:

   ```bash
   tail -15 runs/school-e1-r00.log
   ```

   ```bash
   cat runs/school/e1/r00/summary.json
   ```

2. Recheck the buffer and the held-out levels. Write this into a file in your scratchpad, for
   example `check_e1.py`, and run it from the repository root with
   `.venv/Scripts/python <path>/check_e1.py`:

   ```python
   import glob, json, sys
   sys.path.insert(0, "train")
   import goblin_eval

   rows = [json.loads(l) for l in open("runs/school/e1/buffer.jsonl")]
   items = [{"tokens": r["tokens"], "plans": r["plans"]} for r in rows]
   res = []
   for i in range(0, len(items), 128):
       res += goblin_eval.check(items[i : i + 128], mode="conservative")
   print(len(rows), "buffer levels,", len(res), "plans,", sum(x["verdict"] == "solved" for x in res), "solved")
   held = set()
   for f in glob.glob("data/goblin/test-*.jsonl") + ["data/goblin/probe.jsonl"]:
       for l in open(f):
           i = json.loads(l).get("identity")
           if i:
               held.add(i)
   pool = [json.loads(l) for l in open("runs/school/e1/r00/pool.jsonl")]
   print("held-out identities: buffer", sum(r["identity"] in held for r in rows),
         "pool", sum(r["identity"] in held for r in pool))
   ```

   **Pass:**
   - every plan is solved (the two plan counts are equal);
   - both held-out counts are 0.

   If either fails, say so first in the report. Don't fix or delete anything.

## 4. Report to the user

Give, in this order:

1. Whether it finished, and the total time in hours.
2. The summary table from the log:
   - solved greedy, by a sample, by prefix;
   - liberal only;
   - solved levels, and how many of them beyond the Rust search;
   - newly solved, buffer levels, plans kept.
3. The time of each step: attempt, prefix, liberal, polish.
4. The check from step 3: plans solved of plans, and the held-out counts.
5. Comparison: the 256-level verify rounds solved 10% (r00) and 7% (r01) at k=8. Say what share
   this round solved at k=32. Don't claim Goblin improved: it is the same model, with more samples
   per level.
6. The next step is guarded training on the buffer: step C in
   `docs/goblin-school-next-2026-10-29.md`. It needs new code and two decisions from the user:
   - how large a share of the training mix the buffer should be;
   - whether `run.py` should get a constrained-eval guard.

   Ask; don't start it.
