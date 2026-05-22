# AutoResearch: Diffusion Denoiser for BCI Trajectory Decoding

## Goal

Minimise **imagery-regime trajectory RMSE** by modifying `denoiser.py`.

The MLP decoder (`prepare.py`) is **frozen** — it was trained on execution-regime
neural data and is never touched.  Only `denoiser.py` is editable.

Primary metric to minimise (lower is better):
```
results["primary"]   — mean RMSE on imagery-regime test set after denoising
```

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag** — propose something like `may21`. The branch
   `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `prepare.py` — frozen pipeline, do not modify.
   - `denoiser.py` — the only file you edit.
   - `program.md` — this file (agent instructions).
4. **Verify setup**: Check that `~/.cache/motor-imagery-autoresearch/` contains
   `trajectories.npy` and `mlp_decoder.pt`. If not, tell the human to run:
   ```
   uv run prepare.py
   ```
5. **Initialise results.tsv**: must have the header row (already present if setup
   was followed). The baseline run will populate the first data row.
6. **Confirm and go**.

---

## What you are learning

Whether a generative prior trained on natural hand movements can correct
noisy imagery-decoded trajectories.  The hypothesis: motor imagery signals
produce decoder outputs that are near-manifold perturbations of natural
movement trajectories, and the diffusion denoiser projects them back onto
the manifold.

If the denoiser helps *specifically* in the imagery regime (not equally in
execution), and if that improvement scales with training data volume, that
is a neuroscaling law — the publishable result.

---

## What you CAN modify

Everything inside `denoiser.py`:

- Model architecture (diffusion, flow-matching, VAE, learned Gaussian, …)
- Noise schedule (linear, cosine, or learned)
- Number of inference steps (more steps = better quality but higher latency)
- Conditioning strategy (unconditional vs. conditioned on partial trajectory)
- Temporal window size (how many past steps the model sees)
- Training objective (DDPM-style ε-prediction, x₀-prediction, flow-matching)
- SDEdit noise level (sde_t0 fraction)
- Batch size, learning rate, depth, channel width

## What you CANNOT modify

- `prepare.py` — frozen.  Includes the forward model, MLP decoder, and `evaluate()`.
- The noise model inside `trajectory_to_neural()`.
- The test-set split (fixed seed in `prepare.py`).
- Any guardrail threshold.

---

## Hypotheses to explore (in order)

1. **Unconditional DDPM baseline** — establish the floor.  This is already
   implemented in `denoiser.py`; your first run records it.
2. **Flow matching vs. DDPM** — flow matching typically needs fewer NFE for
   the same quality; can it stay within the 50 ms latency guardrail with
   higher quality?
3. **Conditioning on partial trajectory** — past positions as context.
   Does the model generalise better when it sees trajectory history?
4. **Single-step vs. multi-step inference** — can a distilled single-step
   model (consistency model / flow-matching with 1 step) match multi-step
   quality while staying fast?
5. **Temporal window size** — how many past timesteps actually matter?
6. **Is it learning the manifold or just smoothing?** — compare to Gaussian
   LP filter baseline; the `not_trivial` guardrail checks this automatically.

---

## Experimentation loop

Each experiment runs on a fixed TIME_BUDGET of 600 seconds (10 minutes).
Launch: `uv run denoiser.py > run.log 2>&1`

After the script finishes it prints a summary like:

```
---
primary_rmse:     0.012345
raw_baseline:     0.018000
improvement_pct:  31.42%
smoothness:       0.9200
latency_ms:       18.3
guardrails:       {'denoised_beats_raw': True, ...}
all_pass:         True
---
```

Extract the key metric:
```
grep "^primary_rmse:" run.log
grep "^all_pass:" run.log
```

**LOOP FOREVER:**

1. Check current git state (branch, last commit).
2. Modify `denoiser.py` with an experimental idea.
3. `git commit` the change.
4. Run: `uv run denoiser.py > run.log 2>&1`
5. Read results: `grep "^primary_rmse:\|^all_pass:\|^guardrails:" run.log`
6. If empty / crash: `tail -50 run.log` to debug; fix and re-run or skip.
7. Log to `results.tsv`.
8. If `primary_rmse` improved **AND** `all_pass: True` → keep commit, advance.
9. Otherwise → `git reset --hard HEAD~1` and try something else.

**NEVER STOP.** Do not ask the human if you should continue.  The loop runs
until manually interrupted.  If you run out of ideas, re-read the hypotheses,
try combining near-misses, or attempt a more radical architectural change.

**Timeout:** If a run exceeds 15 minutes, kill it and treat as a crash.

---

## Keep / discard rule

| Condition                                 | Action  |
|-------------------------------------------|---------|
| RMSE improved AND all guardrails pass     | keep    |
| RMSE did not improve                      | discard |
| Any guardrail failed                      | discard |
| `not_trivial` failed (≈ Gaussian filter)  | discard |
| Run crashed                               | discard |

---

## results.tsv format

Tab-separated, NOT comma-separated.  Columns:

```
commit  primary_rmse  raw_baseline  improvement_pct  smoothness  latency_ms  status  description
```

Example:
```
commit	primary_rmse	raw_baseline	improvement_pct	smoothness	latency_ms	status	description
a1b2c3d	0.018000	0.018000	0.00	0.9100	0.5	keep	baseline (no denoiser)
b2c3d4e	0.012345	0.018000	31.42	0.9200	18.3	keep	unconditional DDPM, 20 DDIM steps
c3d4e5f	0.012100	0.018000	32.78	0.9300	22.1	keep	flow matching, 10 steps
```

---

## What to avoid

- Do not change how trajectories are loaded or split (fixed seed in prepare.py).
- Do not change the MLP decoder.
- Do not change the noise model in `trajectory_to_neural()`.
- If the smoothness guardrail fails, add a smoothing loss term — do not relax the guardrail.
- Do not install new packages; only use what is in `pyproject.toml`.
- Do not attempt to add data augmentation that changes the trajectory statistics
  (that would violate the frozen pipeline contract).

---

## Specificity check (run alongside each experiment)

After each `denoiser.py` run, the script also evaluates on the **execution**
regime and prints `execution primary_rmse`.  A good denoiser should improve
the imagery regime substantially while having a smaller or zero effect on
execution (which is already near-manifold).  If execution improves as much
as imagery, the mechanism is just smoothing — not manifold projection.

---

## Notes on compute

- Default device: MPS (Apple Silicon) → CPU fallback.
- `TIME_BUDGET = 600` seconds for training.
- Latency guardrail: < 50 ms per inference call.
- With n_infer=20 DDIM steps the baseline runs in ~15–20 ms on MPS.
  Reducing n_infer speeds inference but may hurt quality.
