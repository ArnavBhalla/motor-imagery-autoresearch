"""
Experiment 18: PCA manifold projection, K=15 components.

More aggressive projection than K=30 — keeps only the dominant 15 modes.
For GRAB data these likely capture the broad motion arc (reach, grasp shape,
lift direction) while discarding fine-grained finger/wrist noise.
"""

import time
import numpy as np
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate)


class PCADenoiser:
    def __init__(self, n_components=15):
        self.n_components = n_components
        self._mean = None
        self._std = None
        self._components = None
        self.trained = False

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        arr = np.stack(trajectories).astype(np.float32)
        N, T, C = arr.shape
        D = T * C
        X = arr.reshape(N, D)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std
        _, _, Vt = np.linalg.svd(X_norm, full_matrices=False)
        self._components = Vt[:self.n_components]
        self.trained = True
        if verbose:
            print(f"PCA done: N={N}, K={self.n_components}, D={D}, {time.perf_counter()-t0:.2f}s")

    def __call__(self, decoded_traj):
        if not self.trained:
            return decoded_traj
        x = decoded_traj.astype(np.float32).reshape(-1)
        x_norm = (x - self._mean) / self._std
        z = self._components @ x_norm
        x_recon_norm = self._components.T @ z
        x_recon = x_recon_norm * self._std + self._mean
        return x_recon.reshape(SEQ_LEN, TRAJ_DIM)


def build_denoiser() -> PCADenoiser:
    return PCADenoiser(n_components=15)


def _log_results(results, description, git_hash="xxxxxxx"):
    import pathlib
    row = "\t".join([git_hash[:7], f"{results['primary']:.6f}", f"{results['raw_baseline']:.6f}",
                     f"{results['improvement_pct']:.2f}", f"{results['smoothness']:.4f}",
                     f"{results['latency_ms']:.1f}",
                     "keep" if results["all_guardrails_pass"] else "discard", description])
    with open(pathlib.Path("results.tsv"), "a") as f: f.write(row + "\n")


if __name__ == "__main__":
    trajs = get_trajectories(); train_t, val_t, test_t = split_trajectories(trajs)
    decoder = load_or_train_decoder(train_t); denoiser = build_denoiser()
    print(f"Training PCA denoiser (K=15) ...")
    denoiser.train(train_t, time_budget=TIME_BUDGET)
    print("\nEvaluating on test set (imagery regime) ...")
    results = evaluate(decoder, denoiser, test_t, snr="imagery")
    print(f"\n---\nprimary_rmse:     {results['primary']:.6f}\nraw_baseline:     {results['raw_baseline']:.6f}")
    print(f"improvement_pct:  {results['improvement_pct']:.2f}%\nsmoothness:       {results['smoothness']:.4f}")
    print(f"latency_ms:       {results['latency_ms']:.1f}\nguardrails:       {results['guardrails']}")
    print(f"all_pass:         {results['all_guardrails_pass']}")
    exec_r = evaluate(decoder, denoiser, test_t, snr="execution")
    print(f"execution primary_rmse: {exec_r['primary']:.6f}  (improvement: {exec_r['improvement_pct']:.2f}%)\n---")
    try:
        import subprocess
        git_hash = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: git_hash = "xxxxxxx"
    _log_results(results, "PCA manifold projection K=15", git_hash)
