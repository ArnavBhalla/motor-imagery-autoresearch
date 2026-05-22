"""
Scaling ablation — run AFTER the main AutoResearch loop has converged.

Tests whether denoiser improvement follows a log-linear (neuroscaling) law:
    RMSE = a - b * ln(hours)

Run this once with the best denoiser architecture found by the agent:
    uv run scaling_ablation.py

Results are saved to scaling_results.tsv and a plot to scaling_curve.png.
"""

import importlib
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from prepare import (
    get_trajectories, split_trajectories,
    load_or_train_decoder,
    sample_trajectory_data, evaluate,
    TIME_BUDGET, SEQ_LEN, SAMPLE_RATE,
    CACHE_DIR,
)

# ---------------------------------------------------------------------------
# Data volumes to probe (hours of trajectory data)
# ---------------------------------------------------------------------------

DATA_VOLUMES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# Train budget per point (shorter than main loop to make ablation tractable)
ABLATION_BUDGET = 300   # 5 minutes per data volume


def train_best_architecture(train_data: list, budget: int) -> object:
    """
    Import and instantiate the current best denoiser from denoiser.py,
    train on train_data for `budget` seconds, and return trained denoiser.
    """
    import denoiser as dn
    importlib.reload(dn)          # pick up any agent edits
    d = dn.build_denoiser()
    d.train(train_data, time_budget=budget, verbose=False)
    return d


def fit_log_linear(hours: np.ndarray, rmse: np.ndarray):
    """
    Fit RMSE = a - b * ln(hours) via ordinary least squares.
    Returns (a, b, r_squared).
    """
    ln_h = np.log(hours)
    # design matrix [1, ln(h)]
    A    = np.column_stack([np.ones_like(ln_h), ln_h])
    coef, _, _, _ = np.linalg.lstsq(A, rmse, rcond=None)
    a, b = coef[0], coef[1]
    pred = a + b * ln_h
    ss_res = np.sum((rmse - pred) ** 2)
    ss_tot = np.sum((rmse - rmse.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return float(a), float(-b), float(r2)   # return b positive (RMSE decreases)


def main():
    print("Scaling ablation: loading data ...")
    all_trajs = get_trajectories()
    train_t, val_t, test_t = split_trajectories(all_trajs)
    decoder    = load_or_train_decoder(train_t)

    max_available_hours = len(train_t) * (SEQ_LEN / SAMPLE_RATE) / 3600.0
    print(f"Max available training data: {max_available_hours:.2f} hours "
          f"({len(train_t)} clips)\n")

    # filter volumes to what we actually have
    volumes = [h for h in DATA_VOLUMES if h <= max_available_hours]
    if len(volumes) < 3:
        # add a few fractional points so the curve still has shape
        clip_s   = SEQ_LEN / SAMPLE_RATE
        max_h    = len(train_t) * clip_s / 3600.0
        volumes  = [max_h * f for f in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]]
    print(f"Volumes to probe: {[f'{h:.2f}h' for h in volumes]}\n")

    results = []
    for hours in volumes:
        subset = sample_trajectory_data(hours, train_t)
        print(f"Volume {hours:.2f} h → {len(subset)} clips — training ...")
        t0 = time.perf_counter()
        denoiser = train_best_architecture(subset, ABLATION_BUDGET)
        elapsed  = time.perf_counter() - t0

        res = evaluate(decoder, denoiser, test_t, snr="imagery")
        row = {
            "hours":          hours,
            "n_clips":        len(subset),
            "primary_rmse":   res["primary"],
            "raw_baseline":   res["raw_baseline"],
            "improvement_pct": res["improvement_pct"],
            "all_pass":       res["all_guardrails_pass"],
            "train_s":        elapsed,
        }
        results.append(row)
        print(f"  RMSE={res['primary']:.5f}  improve={res['improvement_pct']:.1f}%  "
              f"pass={res['all_guardrails_pass']}  ({elapsed:.0f}s)\n")

    # ---- save results -------------------------------------------------------
    df = pd.DataFrame(results)
    tsv_path = Path("scaling_results.tsv")
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"Results saved → {tsv_path}")

    # ---- fit log-linear law -------------------------------------------------
    h_arr    = df["hours"].values
    rmse_arr = df["primary_rmse"].values

    if len(h_arr) >= 3:
        a, b, r2 = fit_log_linear(h_arr, rmse_arr)
        print(f"\nFitted: RMSE = {a:.5f} - {b:.5f} * ln(hours)")
        print(f"R² = {r2:.4f}  {'✓ log-linear scaling law!' if r2 > 0.95 else '— not yet log-linear'}")
    else:
        a, b, r2 = None, None, None
        print("Not enough data points to fit log-linear law.")

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(h_arr, rmse_arr, zorder=5, color="steelblue", label="measured")
    ax.axhline(df["raw_baseline"].iloc[0], linestyle="--", color="gray",
               label="raw baseline (no denoiser)")

    if r2 is not None:
        h_fine = np.logspace(np.log10(h_arr.min()), np.log10(h_arr.max()), 200)
        ax.plot(h_fine, a - b * np.log(h_fine), color="coral",
                label=f"fit: {a:.4f} − {b:.4f}·ln(h)  R²={r2:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Training data (hours)")
    ax.set_ylabel("Imagery RMSE (m)")
    ax.set_title("Neuroscaling law: denoiser improvement vs. trajectory data volume")
    ax.legend()
    fig.tight_layout()
    plot_path = Path("scaling_curve.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
