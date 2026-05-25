"""
Experiment r28: Small supervised CNN denoiser on real ECoG.

Smaller model (ch=64, d=4) better suited to the ~1900-window training set.
Reduced capacity → less overfitting risk on real data.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_bciiv4 import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                              load_dataset, split_dataset,
                              load_or_train_decoder, evaluate)


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
    def __init__(self, channels=64, depth=4, lr=5e-4, batch_size=64, device=None):
        self.device     = device or ("mps"  if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu")
        self.model      = DenoiseCNN(channels=channels, depth=depth).to(self.device)
        self.opt        = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self._mean      = self._std = None
        self.trained    = False
        self.batch_size = batch_size

    def train(self, decoded_trajs, true_trajs, decoder,
              time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        if verbose:
            print(f"  Building {len(decoded_trajs)} pairs ...")
        X = np.stack(decoded_trajs).astype(np.float32)
        y = np.stack(true_trajs).astype(np.float32)

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X_n = (X - self._mean) / self._std
        y_n = (y - self._mean) / self._std

        Xt = torch.from_numpy(X_n.transpose(0, 2, 1)).to(self.device)
        yt = torch.from_numpy(y_n.transpose(0, 2, 1)).to(self.device)
        N  = len(Xt)
        self.model.train()
        step = epoch = 0
        total_loss = 0.0

        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget:
                    break
                xb = Xt[perm[i:i + self.batch_size]]
                yb = yt[perm[i:i + self.batch_size]]
                loss = F.mse_loss(self.model(xb), yb)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss += loss.item(); step += 1
            epoch += 1
            if time.perf_counter() - t0 >= time_budget:
                break
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | "
                      f"loss {total_loss/step:.5f} | "
                      f"elapsed {time.perf_counter()-t0:.0f}s")

        self.model.eval(); self.trained = True
        if verbose:
            print(f"Training done: {epoch} epochs, {step} steps, "
                  f"{time.perf_counter()-t0:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained:
            return decoded_traj
        x  = (decoded_traj.astype(np.float32) - self._mean[0]) / self._std[0]
        xt = torch.from_numpy(x.T).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(xt)
        return (out.squeeze(0).cpu().numpy().T * self._std[0]) + self._mean[0]


def build_denoiser():
    return SupervisedDenoiser(channels=64, depth=4)


if __name__ == "__main__":
    desc = "real CNN small ch=64 d=4 ECoG-HG"

    decoded, true_t = load_dataset()
    train_t, val_t, test_t, train_n, val_n, test_n = split_dataset(decoded, true_t)
    print(f"Dataset: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test windows")

    decoder  = load_or_train_decoder(train_t, train_n)
    denoiser = build_denoiser()

    print(f"Training small CNN denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
    denoiser.train(train_t, train_n, decoder, time_budget=TIME_BUDGET)

    results = evaluate(decoder, denoiser, test_t, test_n)

    print(f"\n---")
    print(f"primary_rmse:     {results['primary']:.6f}")
    print(f"raw_baseline:     {results['raw_baseline']:.6f}")
    print(f"improvement_pct:  {results['improvement_pct']:.2f}%")
    print(f"smoothness:       {results['smoothness']:.4f}")
    print(f"latency_ms:       {results['latency_ms']:.1f}")
    print(f"guardrails:       {results['guardrails']}")
    print(f"all_pass:         {results['all_pass']}")
    print(f"---")

    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "xxxxxxx"

    row = "\t".join([git_hash, f"{results['primary']:.6f}",
                     f"{results['raw_baseline']:.6f}",
                     f"{results['improvement_pct']:.2f}",
                     f"{results['smoothness']:.4f}",
                     f"{results['latency_ms']:.1f}",
                     "keep" if results["all_pass"] else "discard",
                     desc])
    with open("results_bciiv4.tsv", "a") as f:
        f.write(row + "\n")
