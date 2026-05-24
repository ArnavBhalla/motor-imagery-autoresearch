"""
Experiment 20: FFT frequency-domain projection.

Trains by estimating the max signal frequency from training trajectories (95th
percentile of spectral centroid). At inference: FFT the decoded trajectory,
zero out all components above the cutoff, IFFT back.

This is a brick-wall low-pass filter — different from Gaussian smoothing
(which has a soft Gaussian roll-off in frequency space) and should pass
the not_trivial guardrail by removing substantially more high-frequency noise
than sigma=2 Gaussian does.
"""

import time
import numpy as np
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate)


class FFTDenoiser:
    def __init__(self, percentile=95):
        self.percentile = percentile
        self._cutoff_bin = None
        self.trained = False

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        arr = np.stack(trajectories).astype(np.float32)  # (N, T, 3)
        N, T, C = arr.shape
        # FFT along time axis for each trajectory and axis
        spectra = np.abs(np.fft.rfft(arr, axis=1))  # (N, T//2+1, 3)
        # For each trajectory+axis, find highest freq bin with >1% of total power
        cutoffs = []
        for i in range(N):
            for c in range(C):
                power = spectra[i, :, c] ** 2
                total = power.sum() + 1e-8
                cumpower = np.cumsum(power) / total
                # last bin where cumulative power < 99% of total
                idx = np.searchsorted(cumpower, 0.99)
                cutoffs.append(int(idx))
        self._cutoff_bin = int(np.percentile(cutoffs, self.percentile)) + 1
        self.trained = True
        if verbose:
            freq_hz = self._cutoff_bin  # 1 bin = 1 Hz at 100Hz/100 pts
            print(f"FFT cutoff: bin {self._cutoff_bin} (~{freq_hz} Hz), {time.perf_counter()-t0:.2f}s")

    def __call__(self, decoded_traj):
        if not self.trained:
            return decoded_traj
        x = decoded_traj.astype(np.float32)  # (T, 3)
        X = np.fft.rfft(x, axis=0)          # (T//2+1, 3)
        X[self._cutoff_bin:] = 0.0
        return np.fft.irfft(X, n=x.shape[0], axis=0).astype(np.float32)


def build_denoiser() -> FFTDenoiser:
    return FFTDenoiser(percentile=95)


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
    print(f"Training FFT denoiser ...")
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
    _log_results(results, "FFT frequency projection p95 cutoff", git_hash)
