"""
Experiment 02: x0-prediction instead of epsilon-prediction.

Hypothesis: predicting x0 directly (rather than noise eps) gives more
stable gradients for trajectory data because trajectories have meaningful
global structure. The model can exploit long-range temporal consistency.

Key change: loss = MSE(model(x_t, t), x0) instead of MSE(model(x_t, t), eps).
Inference: DDIM x0-pred update rule.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
    get_trajectories, split_trajectories,
    load_or_train_decoder, evaluate,
    CACHE_DIR,
)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, t_dim: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1  = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2  = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1  = nn.GroupNorm(4, channels)
        self.norm2  = nn.GroupNorm(4, channels)
        self.t_proj = nn.Linear(t_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h  = F.silu(self.norm1(self.conv1(x)))
        ts = self.t_proj(t_emb).unsqueeze(-1)
        scale, shift = ts.chunk(2, dim=1)
        h  = self.norm2(h) * (1.0 + scale) + shift
        h  = F.silu(self.conv2(h))
        return x + h


class DenoiseNet(nn.Module):
    """Predicts x0 (clean trajectory) from noisy x_t and timestep t."""

    def __init__(self, traj_dim=TRAJ_DIM, channels=64, depth=4, t_dim=32):
        super().__init__()
        self.t_emb    = SinusoidalPositionEmbedding(t_dim)
        self.t_mlp    = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
                                      nn.Linear(t_dim * 2, t_dim))
        self.proj_in  = nn.Conv1d(traj_dim, channels, 1)
        self.blocks   = nn.ModuleList([ResidualBlock1D(channels, t_dim) for _ in range(depth)])
        self.proj_out = nn.Conv1d(channels, traj_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, t):
        h     = self.proj_in(x.permute(0, 2, 1))
        t_emb = self.t_mlp(self.t_emb(t))
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.proj_out(h).permute(0, 2, 1)


def make_linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas     = torch.linspace(beta_start, beta_end, T)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    return {
        "betas":     betas,
        "alpha_bar": alpha_bar,
        "sqrt_ab":   alpha_bar.sqrt(),
        "sqrt_1mab": (1.0 - alpha_bar).sqrt(),
        "alpha_bar_prev": alpha_bar_prev,
    }


class DDPM:
    """x0-prediction DDPM: model predicts clean trajectory directly."""

    def __init__(self, T_diff=100, sde_t0=0.4, n_infer=20,
                 channels=64, depth=4, lr=2e-4, batch_size=256, t_dim=32,
                 device=None):
        self.T_diff    = T_diff
        self.sde_t0    = sde_t0
        self.n_infer   = n_infer
        self.device    = device or self._auto_device()
        self.sched     = {k: v.to(self.device)
                         for k, v in make_linear_schedule(T_diff).items()}
        self.model     = DenoiseNet(channels=channels, depth=depth, t_dim=t_dim).to(self.device)
        self.opt       = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self._mean = self._std = None
        self.trained = False

    @staticmethod
    def _auto_device():
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available():         return "cuda"
        return "cpu"

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        arr = np.stack(trajectories).astype(np.float32)
        self._mean = arr.mean(axis=(0, 1), keepdims=True)
        self._std  = arr.std(axis=(0, 1), keepdims=True) + 1e-8
        arr = (arr - self._mean) / self._std
        dataset = torch.from_numpy(arr).to(self.device)
        N = len(dataset)
        self.model.train()
        step, epoch, total_loss = 0, 0, 0.0
        t_start = time.perf_counter()

        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t_start >= time_budget:
                    break
                x0    = dataset[perm[i:i+self.batch_size]]
                B     = len(x0)
                t_idx = torch.randint(0, self.T_diff, (B,), device=self.device)
                eps   = torch.randn_like(x0)
                sqrt_ab  = self.sched["sqrt_ab"][t_idx].view(B, 1, 1)
                sqrt_1mb = self.sched["sqrt_1mab"][t_idx].view(B, 1, 1)
                x_t   = sqrt_ab * x0 + sqrt_1mb * eps

                # x0-prediction: model predicts clean trajectory
                pred_x0 = self.model(x_t, t_idx.float())
                loss = F.mse_loss(pred_x0, x0)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss += loss.item()
                step += 1

            epoch += 1
            if time.perf_counter() - t_start >= time_budget:
                break
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | "
                      f"loss {total_loss/step:.5f} | "
                      f"elapsed {time.perf_counter()-t_start:.0f}s")

        self.model.eval()
        self.trained = True
        if verbose:
            print(f"Training done: {epoch} epochs, {step} steps, "
                  f"{time.perf_counter()-t_start:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained:
            return decoded_traj
        x = (decoded_traj - self._mean[0]) / self._std[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        t0       = int(self.sde_t0 * self.T_diff)
        eps0     = torch.randn_like(x_t)
        sqrt_ab  = self.sched["sqrt_ab"][t0].item()
        sqrt_1mb = self.sched["sqrt_1mab"][t0].item()
        x_t = sqrt_ab * x_t + sqrt_1mb * eps0

        ts = torch.linspace(t0, 0, self.n_infer + 1, dtype=torch.long,
                            device=self.device).clamp(0, self.T_diff - 1)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_infer):
                t_cur  = ts[i].unsqueeze(0)
                t_prev = ts[i+1].unsqueeze(0)
                ab_cur  = self.sched["alpha_bar"][t_cur].view(1, 1, 1)
                ab_prev = self.sched["alpha_bar"][t_prev].view(1, 1, 1)

                # x0-prediction: model outputs x0 directly
                pred_x0 = self.model(x_t, t_cur.float()).clamp(-3, 3)
                # DDIM update via x0
                pred_eps = (x_t - ab_cur.sqrt() * pred_x0) / (1.0 - ab_cur).sqrt()
                x_t = ab_prev.sqrt() * pred_x0 + (1.0 - ab_prev).sqrt() * pred_eps

        out = x_t.squeeze(0).cpu().numpy()
        return out * self._std[0] + self._mean[0]


def build_denoiser() -> DDPM:
    return DDPM(T_diff=100, sde_t0=0.4, n_infer=20, channels=64, depth=4, lr=2e-4, batch_size=256)


def _log_results(results, description, git_hash="xxxxxxx"):
    import pathlib
    tsv = pathlib.Path("results.tsv")
    row = "\t".join([git_hash[:7], f"{results['primary']:.6f}",
                     f"{results['raw_baseline']:.6f}", f"{results['improvement_pct']:.2f}",
                     f"{results['smoothness']:.4f}", f"{results['latency_ms']:.1f}",
                     "keep" if results["all_guardrails_pass"] else "discard", description])
    with open(tsv, "a") as f:
        f.write(row + "\n")


if __name__ == "__main__":
    trajs = get_trajectories()
    train_t, val_t, test_t = split_trajectories(trajs)
    decoder  = load_or_train_decoder(train_t)
    denoiser = build_denoiser()
    print(f"Training denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
    denoiser.train(train_t, time_budget=TIME_BUDGET)
    print("\nEvaluating on test set (imagery regime) ...")
    results = evaluate(decoder, denoiser, test_t, snr="imagery")
    print(f"\n---")
    print(f"primary_rmse:     {results['primary']:.6f}")
    print(f"raw_baseline:     {results['raw_baseline']:.6f}")
    print(f"improvement_pct:  {results['improvement_pct']:.2f}%")
    print(f"smoothness:       {results['smoothness']:.4f}")
    print(f"latency_ms:       {results['latency_ms']:.1f}")
    print(f"guardrails:       {results['guardrails']}")
    print(f"all_pass:         {results['all_guardrails_pass']}")
    exec_r = evaluate(decoder, denoiser, test_t, snr="execution")
    print(f"execution primary_rmse: {exec_r['primary']:.6f}  (improvement: {exec_r['improvement_pct']:.2f}%)")
    print("---")
    try:
        import subprocess
        git_hash = subprocess.check_output(["git","rev-parse","--short","HEAD"],
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "xxxxxxx"
    _log_results(results, "x0-prediction DDPM (SDEdit, 20 DDIM steps)", git_hash)
