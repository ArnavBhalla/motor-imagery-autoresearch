"""
Experiment 25: Supervised CNN with augmented training pairs (n_aug=4).

Instead of 1 noise realization per training trajectory, generate 4 different
imagery noise seeds per clean trajectory → 4× more training pairs (6400 total).
Each realization sees different noise, so the CNN learns to correct the
decoder's systematic errors rather than memorizing specific noise patterns.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET, FIXED_SEED,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate,
                     trajectory_to_neural)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel=9):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        return x + self.norm2(self.conv2(h))


class DenoiseCNN(nn.Module):
    def __init__(self, dim=TRAJ_DIM, channels=64, depth=4, kernel=9):
        super().__init__()
        self.proj_in  = nn.Conv1d(dim, channels, 1)
        self.blocks   = nn.ModuleList([ResBlock(channels, kernel) for _ in range(depth)])
        self.proj_out = nn.Conv1d(channels, dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h)
        return x + self.proj_out(h)


class SupervisedDenoiser:
    def __init__(self, n_aug=4, channels=64, depth=4, lr=1e-3, batch_size=128, device=None):
        self.n_aug = n_aug
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenoiseCNN(channels=channels, depth=depth).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._mean = self._std = None
        self.trained = False
        self.batch_size = batch_size

    def _make_pairs(self, trajectories, decoder):
        rng = np.random.RandomState(FIXED_SEED + 1)
        X_list, y_list = [], []
        for traj in trajectories:
            for _ in range(self.n_aug):
                seed = int(rng.randint(1_000_000))
                neural = trajectory_to_neural(traj, snr="imagery", seed=seed)
                decoded = decoder(neural)
                X_list.append(decoded)
                y_list.append(traj)
        return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        decoder = load_or_train_decoder(trajectories)
        if verbose:
            print(f"  Generating {len(trajectories)}×{self.n_aug} augmented pairs ...")
        X, y = self._make_pairs(trajectories, decoder)
        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X_n = (X - self._mean) / self._std
        y_n = (y - self._mean) / self._std
        Xt = torch.from_numpy(X_n.transpose(0, 2, 1)).to(self.device)
        yt = torch.from_numpy(y_n.transpose(0, 2, 1)).to(self.device)
        N = len(Xt)
        self.model.train()
        step = epoch = 0; total_loss = 0.0
        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget: break
                xb = Xt[perm[i:i+self.batch_size]]
                yb = yt[perm[i:i+self.batch_size]]
                loss = F.mse_loss(self.model(xb), yb)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); total_loss += loss.item(); step += 1
            epoch += 1
            if time.perf_counter() - t0 >= time_budget: break
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | loss {total_loss/step:.5f} | "
                      f"elapsed {time.perf_counter()-t0:.0f}s")
        self.model.eval(); self.trained = True
        if verbose:
            print(f"Training done: {epoch} epochs, {step} steps, {time.perf_counter()-t0:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained: return decoded_traj
        x = (decoded_traj.astype(np.float32) - self._mean[0]) / self._std[0]
        xt = torch.from_numpy(x.T).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(xt)
        return (out.squeeze(0).cpu().numpy().T * self._std[0]) + self._mean[0]


def build_denoiser() -> SupervisedDenoiser:
    return SupervisedDenoiser(n_aug=4, channels=64, depth=4, lr=1e-3, batch_size=128)


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
    print(f"Training augmented supervised CNN for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "supervised CNN augmented n_aug=4 channels=64 depth=4", git_hash)
