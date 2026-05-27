# autoresearch program — autonomous loop instructions

This file is the agent's protocol. Read each loop iteration.

**Monorepo note:** Stage only files inside this project's `autoresearch/` and the configured `edit_files`. NEVER `git add -A`.

## Setup (one-time)

1. Read `autoresearch/config.yaml` — train_cmd, metric, edit_files.
2. Read `edit_files` in full to understand scope.
3. Read `readonly_files` for context but NEVER modify.
4. Confirm baseline exists in `results.tsv`. If empty/missing, run baseline first :
   - `eval "$train_cmd" > autoresearch/run.log 2>&1`
   - Extract metric, log to results.tsv as `keep / baseline`
   - Commit results.tsv.

## Loop

LOOP FOREVER (until user interrupts or `--max N` reached) :

1. **Inspect git state** : current branch, last commit, last metric value (best so far).
2. **Generate experimental idea** : reason about what to try next.
   - Use `idea_hints` from config if present.
   - Avoid repeating ideas already discarded (check results.tsv).
   - Mix small tweaks (LR, batch) with bigger structural changes (architecture, optimizer).
3. **Edit ONE or more `edit_files`** with the idea. Keep diff minimal and reviewable.
4. **Commit** : `git add <edit_files> && git commit -m "experiment: <short description>"`
5. **Run** : `eval "$train_cmd" > autoresearch/run.log 2>&1`
   - REDIRECT all output to file. NEVER tee. NEVER let log flood context.
6. **Extract metric** : `grep -oE "$metric_pattern" autoresearch/run.log | tail -1`
7. **Decision** :
   - **Improved** (per `metric_direction`) :
     - status = `keep`
     - Append row to `results.tsv` : `<commit>\t<metric>\tkeep\t<description>`
     - `git add autoresearch/results.tsv && git commit --amend --no-edit`
   - **Worse or equal** :
     - status = `discard`
     - Append row to `results.tsv` BEFORE reverting (with the discarded commit hash)
     - Commit results.tsv to last-kept commit : tricky, see "Discard protocol" below
     - `git reset --hard <last_keep_commit>`
   - **Crash (grep empty)** :
     - `tail -n 50 autoresearch/run.log` to read trace
     - If trivial (typo, import, off-by-one) : fix, amend commit, retry ONCE
     - Else : status = `crash`, log to results.tsv, revert
8. **Loop back to step 1**.

## Discard protocol (handling results.tsv across reverts)

The agent should accumulate ALL experiment rows in results.tsv, including discards. Reverting via `git reset --hard` loses the row. Solution :

- Before `git reset --hard <last_keep>`, copy results.tsv contents to memory.
- After reset, restore the row, `git add autoresearch/results.tsv && git commit --amend --no-edit` to fold it into the last_keep commit.

Alternative : keep results.tsv on a separate branch, but that complicates the loop. Use the copy-restore approach by default.

## Output format expected from `$train_cmd`

Standardized footer for parseability :

```
---
<metric_name>:    <value>
<extra_metric_1>: <value>
<extra_metric_2>: <value>
training_seconds: <value>
```

If your train script doesn't output this, modify it to do so (this is a one-time setup, not part of the loop).

## What you CAN do

- Modify `edit_files` (architecture, optimizer, hyperparameters, training loop logic).
- Reorganize logic within `edit_files`.
- Read any file in repo for context.

## What you CANNOT do

- Modify `readonly_files` or anything outside `edit_files`.
- Install new packages (no `uv add`, no `pip install`). Use only what's in pyproject.toml / requirements.txt.
- Change the metric or evaluation harness.
- `git push` to remote (unless user explicit request).
- Skip the redirect : ALL `eval $train_cmd` output goes to `autoresearch/run.log`.

## Simplicity criterion

All else equal, simpler is better. A 0.001 metric gain with 30 lines of hacky code is worse than equal metric with deleted code. Removing complexity AND improving = best outcome.

## Timeout

If `$train_cmd` runs > `2 × time_budget_seconds`, kill it (`pkill -f` or job control). Log as crash. Move on.

## NEVER STOP

Once setup done, do NOT ask "should I continue?". User may be asleep. Loop until manually stopped or `--max N` reached. If you run out of ideas :
1. Re-read `edit_files` slowly for new angles.
2. Look up referenced papers in code comments.
3. Try combining previous near-misses.
4. Try a radical architectural change.
5. Keep going.
