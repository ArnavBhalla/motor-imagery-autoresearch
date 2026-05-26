"""
Experiment 14: Tighter-bottleneck AE manifold projection (bottleneck=8).

AE-32 got 1.37% — the 32D bottleneck is too loose for synthetic sinusoidal
trajectories whose intrinsic dimension is ~10-15D. Compressing to 8D forces
the AE to capture only the dominant modes (amplitude/frequency/phase per axis),
making reconstructions that differ substantially from the original off-manifold
imagery trajectory — and therefore differ from Gaussian smoothing.
"""

import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate)


class TrajectoryAE(nn.Module):
    def __init__(self, traj_dim=TRAJ_DIM, seq_len=SEQ_LEN, bottleneck=8, channels=64):
        super().__init__()
        self.seq_len = seq_len; self.channels = channels
        self.enc = nn.Sequential(
            nn.Conv1d(traj_dim, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
        )
        self.enc_proj = nn.Linear(channels * seq_len, bottleneck)
        self.dec_proj = nn.Linear(bottleneck, channels * seq_len)
        self.dec = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.SiLU(),
            nn.Conv1d(channels, traj_dim, 5, padding=2),
        )

    def forward(self, x):
        h = self.enc(x.permute(0, 2, 1))
        z = self.enc_proj(h.flatten(1))
        h2 = self.dec_proj(z).view(-1, self.channels, self.seq_len)
        return self.dec(h2).permute(0, 2, 1)


class AEDenoiser:
    def __init__(self, bottleneck=8, channels=64, lr=1e-3, batch_size=256, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryAE(bottleneck=bottleneck, channels=channels).to(self.device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self._mean = self._std = None
        self.trained = False

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        arr = np.stack(trajectories).astype(np.float32)
        self._mean = arr.mean(axis=(0, 1), keepdims=True)
        self._std  = arr.std(axis=(0, 1),  keepdims=True) + 1e-8
        arr = (arr - self._mean) / self._std
        dataset = torch.from_numpy(arr).to(self.device)
        N = len(dataset)
        self.model.train()
        step = epoch = 0; total_loss = 0.0; t0 = time.perf_counter()
        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget: break
                x = dataset[perm[i:i+self.batch_size]]
                loss = F.mse_loss(self.model(x), x)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); total_loss += loss.item(); step += 1
            epoch += 1
            if time.perf_counter() - t0 >= time_budget: break
            if verbose and epoch % 100 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | loss {total_loss/step:.5f} | elapsed {time.perf_counter()-t0:.0f}s")
        self.model.eval(); self.trained = True
        if verbose: print(f"Training done: {epoch} epochs, {step} steps, {time.perf_counter()-t0:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained: return decoded_traj
        x = (decoded_traj - self._mean[0]) / self._std[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_t)
        return out.squeeze(0).cpu().numpy() * self._std[0] + self._mean[0]


def build_denoiser() -> AEDenoiser:
    return AEDenoiser(bottleneck=8, channels=64, lr=1e-3, batch_size=256)


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
    print(f"Training tight-AE denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "tight AE manifold projection bottleneck=8", git_hash)
