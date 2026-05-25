"""
Experiment 32: Ensemble of 5 supervised CNNs (synthetic).

The single CNN (ch=128, d=6) shows high init variance: 15–26% improvement
depending on random seed. Averaging K=5 independent random inits at test time
should reduce variance and consistently approach the top of the distribution.

Each model trains for TIME_BUDGET/K seconds (120s each).
At inference: average outputs of all 5 models.

Novelty: first systematic test of ensembling for post-hoc BCI trajectory correction.
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

K_MODELS = 5   # ensemble size


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
    def __init__(self, dim=TRAJ_DIM, channels=128, depth=6, kernel=9):
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


class EnsembleDenoiser:
    def __init__(self, n_aug=4, channels=128, depth=6, k=K_MODELS,
                 lr=5e-4, batch_size=64, device=None):
        self.n_aug   = n_aug
        self.k       = k
        self.device  = device or ("mps"  if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else "cpu")
        # Each model gets its own random init
        self.models  = [DenoiseCNN(channels=channels, depth=depth).to(self.device)
                        for _ in range(k)]
        self.opts    = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self._mean   = self._std = None
        self.trained = False
        self.batch_size = batch_size

    def _make_pairs(self, trajectories, decoder):
        rng = np.random.RandomState(FIXED_SEED + 1)
        X, y = [], []
        for traj in trajectories:
            for _ in range(self.n_aug):
                seed = int(rng.randint(1_000_000))
                neural = trajectory_to_neural(traj, snr="imagery", seed=seed)
                X.append(decoder(neural))
                y.append(traj)
        return np.stack(X).astype(np.float32), np.stack(y).astype(np.float32)

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        decoder = load_or_train_decoder(trajectories)
        if verbose:
            print(f"  Generating {len(trajectories)}×{self.n_aug} pairs ...")
        X, y = self._make_pairs(trajectories, decoder)

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        X_n = (X - self._mean) / self._std
        y_n = (y - self._mean) / self._std

        Xt = torch.from_numpy(X_n.transpose(0, 2, 1)).to(self.device)
        yt = torch.from_numpy(y_n.transpose(0, 2, 1)).to(self.device)
        N  = len(Xt)

        budget_each = time_budget / self.k
        for idx, (model, opt) in enumerate(zip(self.models, self.opts)):
            t_start = time.perf_counter()
            model.train()
            step = epoch = 0
            total_loss = 0.0
            while True:
                perm = torch.randperm(N, device=self.device)
                for i in range(0, N, self.batch_size):
                    if time.perf_counter() - t_start >= budget_each:
                        break
                    xb = Xt[perm[i:i + self.batch_size]]
                    yb = yt[perm[i:i + self.batch_size]]
                    loss = F.mse_loss(model(xb), yb)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    total_loss += loss.item(); step += 1
                epoch += 1
                if time.perf_counter() - t_start >= budget_each:
                    break
            model.eval()
            if verbose:
                print(f"  Model {idx+1}/{self.k}: {epoch} epochs, "
                      f"{step} steps, loss {total_loss/max(step,1):.5f}, "
                      f"{time.perf_counter()-t_start:.0f}s")

        self.trained = True
        if verbose:
            print(f"Ensemble trained in {time.perf_counter()-t0:.1f}s total")

    def __call__(self, decoded_traj):
        if not self.trained:
            return decoded_traj
        x  = (decoded_traj.astype(np.float32) - self._mean[0]) / self._std[0]
        xt = torch.from_numpy(x.T).float().unsqueeze(0).to(self.device)
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(xt).squeeze(0).cpu().numpy().T
            outputs.append(out)
        avg = np.mean(outputs, axis=0)
        return (avg * self._std[0]) + self._mean[0]


def build_denoiser():
    return EnsembleDenoiser(n_aug=4, channels=128, depth=6, k=K_MODELS)


def _log_results(results, description, git_hash="xxxxxxx"):
    import pathlib
    row = "\t".join([git_hash[:7], f"{results['primary']:.6f}",
                     f"{results['raw_baseline']:.6f}",
                     f"{results['improvement_pct']:.2f}",
                     f"{results['smoothness']:.4f}",
                     f"{results['latency_ms']:.1f}",
                     "keep" if results["all_guardrails_pass"] else "discard",
                     description])
    with open(pathlib.Path("results.tsv"), "a") as f:
        f.write(row + "\n")


if __name__ == "__main__":
    trajs = get_trajectories()
    train_t, val_t, test_t = split_trajectories(trajs)
    decoder  = load_or_train_decoder(train_t)
    denoiser = build_denoiser()
    print(f"Training {K_MODELS}-model CNN ensemble for {TIME_BUDGET}s on {denoiser.device} ...")
    denoiser.train(train_t, time_budget=TIME_BUDGET)
    print("\nEvaluating ...")
    results = evaluate(decoder, denoiser, test_t, snr="imagery")
    print(f"\n---\nprimary_rmse:     {results['primary']:.6f}")
    print(f"raw_baseline:     {results['raw_baseline']:.6f}")
    print(f"improvement_pct:  {results['improvement_pct']:.2f}%")
    print(f"smoothness:       {results['smoothness']:.4f}")
    print(f"latency_ms:       {results['latency_ms']:.1f}")
    print(f"guardrails:       {results['guardrails']}")
    print(f"all_pass:         {results['all_guardrails_pass']}")
    exec_r = evaluate(decoder, denoiser, test_t, snr="execution")
    print(f"execution primary_rmse: {exec_r['primary']:.6f}  "
          f"(improvement: {exec_r['improvement_pct']:.2f}%)\n---")
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "xxxxxxx"
    _log_results(results, f"ensemble CNN K={K_MODELS} ch=128 d=6 n_aug=4", git_hash)
