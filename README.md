# femto-gpt

This repository is a code-golf competition around a tiny GPT trainer/inference script in pure Python.

The target is not only to train and sample correctly, but to preserve the *exact observable behavior* of the reference implementation while shrinking source size.

## Goal

The project has three implementations:

1. `microgpt.py` - canonical reference implementation.
2. `femtogpt.py` - shortest valid implementation (strict parity target).
3. `picogpt.py` - rival implementation.

To be considered fully valid in this repo, an implementation must match `microgpt.py` output as a complete log, not only selected metrics.

## What the scripts do

Each script:

1. Loads `input.txt` as the dataset of names/tokens.
2. Builds a tiny character-level GPT-like model with custom scalar autograd.
3. Trains for 1000 steps with Adam.
4. Prints per-step loss lines.
5. Runs inference and prints 20 samples.

The model is intentionally minimal and deterministic (fixed random seed), so exact output matching is possible.

## Files

1. `microgpt.py`: readable baseline, full comments, explicit architecture.
2. `femtogpt.py`: compressed standalone implementation using Python metaprogramming (`exec` over compressed payload).
3. `picogpt.py`: heavily golfed rival implementation.
4. `compare_runs.sh`: benchmark/equivalence harness.
5. `run_results/<timestamp>/`: logs, extracted traces, diffs, and summary.

## Benchmark and validation protocol

`compare_runs.sh` executes all three scripts and computes:

1. Exit status and elapsed time.
2. Model metadata capture from logs:
   1. `num docs:`
   2. `vocab size:`
   3. `num params:`
3. Training trace comparison (`step ... loss ...` lines).
4. Inference comparison (`sample ...` lines).
5. Full raw log equality (`cmp -s`) against `microgpt`.

Strict validity in this project is defined by full log equality and not only partial traces.

## Latest verified result

From `run_results/20260215_114425/summary.txt`:

1. `femtogpt vs microgpt`
   1. `full log match: yes`
   2. `loss trace match: yes`
   3. `sample output match: yes`
2. `picogpt vs microgpt`
   1. `full log match: no`
   2. `loss trace match: yes`
   3. `sample output match: yes`

Source sizes:

1. `microgpt.py`: 9165 bytes
2. `picogpt.py`: 1997 bytes
3. `femtogpt.py`: 2530 bytes (1298 characters)

Note: the current `femtogpt.py` is optimized for character count using high-Unicode payload encoding. It is below 1500 characters, but not below 2000 bytes.

## Why `picogpt.py` is not valid under strict parity

`picogpt.py` reproduces the loss trace and sample outputs, but it does not reproduce the full output contract of `microgpt.py`.

The exact diff against `microgpt.log` shows three missing metadata lines at the top:

1. `num docs: 32033`
2. `vocab size: 27`
3. `num params: 4192`

Because those lines are part of the observable program output, `picogpt.py` fails strict equivalence (`full log match: no`), even though core training/inference traces match.

In short:

1. Numerically equivalent training trace: yes.
2. Numerically equivalent sampling trace: yes.
3. Behaviorally identical CLI program output: no.

For this repo's rules, that makes `picogpt.py` non-valid as a drop-in replacement for `microgpt.py`.

## Run it

```bash
./compare_runs.sh
```

Outputs are written to `run_results/<timestamp>/` and summarized in `summary.txt`.
