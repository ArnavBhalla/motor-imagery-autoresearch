# Motor Imagery AutoResearch

Autonomous diffusion denoiser research for BCI trajectory decoding, inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent modifies `denoiser.py`, trains for 10 minutes, checks if the imagery-
regime RMSE improved, keeps or discards, and repeats.  You wake up to a log of
experiments and (hopefully) a better denoiser.

---

## The Research Question

> Can a diffusion-based denoiser — trained exclusively on clean natural hand
> movement trajectories — reduce the performance gap of an MLP neural decoder
> when applied to motor imagery signals?

Motor imagery signals are ~3–5× noisier than executed movement signals.  A
denoiser trained on natural trajectory statistics could project the noisy decoded
trajectories back onto the clean-movement manifold, partially closing the gap.

### Sub-questions

1. Does the denoiser improve imagery-regime accuracy at all? *(existence)*
2. Is the benefit specific to imagery or does it help execution equally? *(specificity)*
3. Does improvement scale log-linearly with pretraining data volume? *(neuroscaling law)*

---

## Project Structure

```
prepare.py          — frozen pipeline (do not modify)
denoiser.py         — the ONLY file the agent edits
program.md          — agent instructions / research org
scaling_ablation.py — scaling law experiment (run post-loop)
results.tsv         — experiment log
pyproject.toml      — dependencies
```

---

## Quick Start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. One-time setup: generate trajectories + train MLP decoder (~2 min)
uv run prepare.py

# 4. Run a single experiment manually (~10 min)
uv run denoiser.py

# 5. Check results
cat results.tsv
```

### With GRAB data (recommended)

```bash
# Clone and extract GRAB wrist trajectories first:
git clone https://github.com/otaheri/GRAB
cd GRAB && pip install -r requirements.txt
python grab/tools/extract_wrist_trajectories.py --output /tmp/grab_wrists

# Then run setup pointing at the extracted files:
uv run prepare.py --grab-dir /tmp/grab_wrists
```

---

## Running AutoResearch

Point your Claude/Codex agent at this repo (disable all permissions) and prompt:

```
Hi, have a look at program.md and let's kick off a new experiment!
Let's do the setup first.
```

The agent will:
1. Create a branch `autoresearch/<tag>`
2. Run setup, verify the execution–imagery gap, establish a baseline
3. Iterate: modify `denoiser.py` → train → evaluate → keep/discard → repeat

---

## Pipeline

```
GRAB / synthetic trajectories
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
trajectory_to_neural(snr='execution')   train denoiser.py  ← agent edits here
        │
        ▼
train MLP decoder (frozen after this)
        │
        ▼
trajectory_to_neural(snr='imagery')
        │
        ▼
MLP decoder → noisy trajectory estimate
        │
        ▼
denoiser.py → cleaned trajectory
        │
        ▼
evaluate() → primary RMSE + guardrails → keep / discard
```

---

## Guardrails

| Guardrail              | Threshold                        |
|------------------------|----------------------------------|
| denoised_beats_raw     | RMSE(denoised) < RMSE(raw)       |
| no_workspace_violation | all trajectory points in bounds  |
| smoothness             | jerk score > 0.85                |
| latency_ms             | < 50 ms per inference call       |
| not_trivial            | not equivalent to Gaussian LP    |

All five must pass for a result to be kept.

---

## Primary Metric

**`primary_rmse`** — mean trajectory RMSE (m) on the imagery-regime held-out test set.
Lower is better.

---

## Scaling Ablation

After the loop converges to the best architecture, run:

```bash
uv run scaling_ablation.py
```

This fits `RMSE = a - b * ln(hours)`.  R² > 0.95 means you have a neuroscaling law.

---

## Day-by-Day Plan

| Day | Task |
|-----|------|
| 1   | Run `uv run prepare.py`, verify gap (should be 40–60% worse in imagery regime) |
| 2   | Run first `uv run denoiser.py`, confirm baseline logs to results.tsv |
| 3   | Start overnight AutoResearch loop |
| 4+  | Review results.tsv, note best architectures, guide agent towards scaling ablation |
