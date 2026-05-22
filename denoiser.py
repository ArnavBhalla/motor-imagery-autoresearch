"""
Denoiser pipeline — the ONLY file you may edit.

Current baseline: unconditional DDPM with a compact 1-D temporal CNN.

Inference strategy: SDEdit — treat the decoded (noisy) trajectory as being
at an intermediate noise level, then run a short reverse chain to project
it back onto the clean-trajectory manifold.

Agent: modify architecture, noise schedule, number of inference steps,
conditioning strategy, temporal window size, training objective, etc.
Keep/discard rule: keep if primary RMSE improves AND all guardrails pass.
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

# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Encodes a scalar diffusion timestep t into a vector."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,)  →  (B, dim)
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    """
    Temporal residual block operating on (B, C, T).
    Injects diffusion-time conditioning via a learned scale/shift.
    """

    def __init__(self, channels: int, t_dim: int, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.conv1  = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2  = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1  = nn.GroupNorm(4, channels)
        self.norm2  = nn.GroupNorm(4, channels)
        self.t_proj = nn.Linear(t_dim, channels * 2)   # → scale, shift

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h  = F.silu(self.norm1(self.conv1(x)))
        ts = self.t_proj(t_emb).unsqueeze(-1)           # (B, 2C, 1)
        scale, shift = ts.chunk(2, dim=1)
        h  = self.norm2(h) * (1.0 + scale) + shift
        h  = F.silu(self.conv2(h))
        return x + h


class DenoiseNet(nn.Module):
    """
    Compact 1-D CNN that predicts the noise added at step t.
    Input : (B, TRAJ_DIM, T) noisy trajectory + diffusion time embedding
    Output: (B, TRAJ_DIM, T) predicted noise
    """

    def __init__(
        self,
        traj_dim: int = TRAJ_DIM,
        seq_len: int  = SEQ_LEN,
        channels: int = 64,
        depth: int    = 4,
        t_dim: int    = 32,
    ):
        super().__init__()
        self.t_emb  = SinusoidalPositionEmbedding(t_dim)
        self.t_mlp  = nn.Sequential(nn.Linear(t_dim, t_dim * 2), nn.SiLU(),
                                    nn.Linear(t_dim * 2, t_dim))
        self.proj_in  = nn.Conv1d(traj_dim, channels, 1)
        self.blocks   = nn.ModuleList([
            ResidualBlock1D(channels, t_dim) for _ in range(depth)
        ])
        self.proj_out = nn.Conv1d(channels, traj_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x : (B, T, D)  →  work in (B, D, T) channel-first
        h     = self.proj_in(x.permute(0, 2, 1))
        t_emb = self.t_mlp(self.t_emb(t))              # (B, t_dim)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.proj_out(h).permute(0, 2, 1)       # (B, T, D)


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def make_linear_schedule(T: int, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> dict:
    betas      = torch.linspace(beta_start, beta_end, T)
    alphas     = 1.0 - betas
    alpha_bar  = torch.cumprod(alphas, dim=0)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    sqrt_ab    = alpha_bar.sqrt()
    sqrt_1mab  = (1.0 - alpha_bar).sqrt()
    # posterior variance for q(x_{t-1} | x_t, x_0)
    post_var   = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
    return {
        "betas":        betas,
        "alphas":       alphas,
        "alpha_bar":    alpha_bar,
        "sqrt_ab":      sqrt_ab,
        "sqrt_1mab":    sqrt_1mab,
        "post_var":     post_var,
    }


# ---------------------------------------------------------------------------
# DDPM denoiser
# ---------------------------------------------------------------------------

class DDPM:
    """
    Unconditional DDPM for trajectory denoising.

    Training: standard DDPM objective — predict noise ε.
    Inference: SDEdit — partially corrupt the decoded trajectory to a
               mid-level noise step, then denoise back to t=0.
    """

    def __init__(
        self,
        T_diff:      int   = 100,   # number of diffusion steps
        sde_t0:      float = 0.4,   # SDEdit start fraction (0.4 × T_diff)
        n_infer:     int   = 20,    # DDIM reverse steps at inference
        channels:    int   = 64,
        depth:       int   = 4,
        lr:          float = 2e-4,
        batch_size:  int   = 256,
        t_dim:       int   = 32,
        device:      str   = None,
    ):
        self.T_diff     = T_diff
        self.sde_t0     = sde_t0
        self.n_infer    = n_infer
        self.device     = device or self._auto_device()

        self.sched      = {k: v.to(self.device)
                          for k, v in make_linear_schedule(T_diff).items()}

        self.model      = DenoiseNet(channels=channels, depth=depth,
                                     t_dim=t_dim).to(self.device)
        self.opt        = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size

        # normalisation stats (set during train)
        self._mean  = None
        self._std   = None
        self.trained = False

    @staticmethod
    def _auto_device():
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, trajectories: list, time_budget: int = TIME_BUDGET,
              verbose: bool = True):
        """
        Train on clean (T, D) position trajectories.
        Runs for exactly `time_budget` wall-clock seconds.
        """
        arr = np.stack(trajectories).astype(np.float32)   # (N, T, D)

        # Normalise to zero-mean unit-std per dimension
        self._mean = arr.mean(axis=(0, 1), keepdims=True)
        self._std  = arr.std(axis=(0, 1), keepdims=True) + 1e-8
        arr        = (arr - self._mean) / self._std

        dataset = torch.from_numpy(arr).to(self.device)   # (N, T, D)
        N       = len(dataset)

        self.model.train()
        step       = 0
        t_start    = time.perf_counter()
        epoch      = 0
        total_loss = 0.0

        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                elapsed = time.perf_counter() - t_start
                if elapsed >= time_budget:
                    break

                x0    = dataset[perm[i : i + self.batch_size]]  # (B, T, D)
                B     = len(x0)
                t_idx = torch.randint(0, self.T_diff, (B,), device=self.device)
                eps   = torch.randn_like(x0)

                sqrt_ab  = self.sched["sqrt_ab"][t_idx].view(B, 1, 1)
                sqrt_1mb = self.sched["sqrt_1mab"][t_idx].view(B, 1, 1)
                x_t      = sqrt_ab * x0 + sqrt_1mb * eps   # forward diffusion

                pred_eps = self.model(x_t, t_idx.float())
                loss     = F.mse_loss(pred_eps, eps)

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                total_loss += loss.item()
                step       += 1

            epoch += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= time_budget:
                break

            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | "
                      f"loss {total_loss/step:.5f} | "
                      f"elapsed {elapsed:.0f}s")

        self.model.eval()
        self.trained = True
        if verbose:
            print(f"Training done: {epoch} epochs, {step} steps, "
                  f"{time.perf_counter() - t_start:.1f}s")

    # ------------------------------------------------------------------
    # Inference (SDEdit)
    # ------------------------------------------------------------------

    def __call__(self, decoded_traj: np.ndarray) -> np.ndarray:
        """
        Denoise a single (T, D) decoded trajectory.
        SDEdit: partially corrupt to noise level t0, then reverse-denoise.
        """
        if not self.trained:
            return decoded_traj   # identity before training

        x = (decoded_traj - self._mean[0]) / self._std[0]   # normalise
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)  # (1, T, D)

        # Forward corrupt to t0
        t0    = int(self.sde_t0 * self.T_diff)
        eps0  = torch.randn_like(x_t)
        sqrt_ab  = self.sched["sqrt_ab"][t0].item()
        sqrt_1mb = self.sched["sqrt_1mab"][t0].item()
        x_t = sqrt_ab * x_t + sqrt_1mb * eps0

        # DDIM-style reverse from t0 → 0
        ts = torch.linspace(t0, 0, self.n_infer + 1, dtype=torch.long,
                            device=self.device).clamp(0, self.T_diff - 1)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_infer):
                t_cur  = ts[i].unsqueeze(0)
                t_prev = ts[i + 1].unsqueeze(0)

                ab_cur  = self.sched["alpha_bar"][t_cur].view(1, 1, 1)
                ab_prev = self.sched["alpha_bar"][t_prev].view(1, 1, 1)

                pred_eps  = self.model(x_t, t_cur.float())
                # DDIM deterministic update
                x0_hat    = (x_t - (1.0 - ab_cur).sqrt() * pred_eps) / ab_cur.sqrt()
                x0_hat    = x0_hat.clamp(-3.0, 3.0)
                x_t       = ab_prev.sqrt() * x0_hat + (1.0 - ab_prev).sqrt() * pred_eps

        out = x_t.squeeze(0).cpu().numpy()
        return out * self._std[0] + self._mean[0]   # denormalise


# ---------------------------------------------------------------------------
# Build & run one experiment
# ---------------------------------------------------------------------------

def build_denoiser() -> DDPM:
    """Construct the denoiser.  Modify hyperparameters here."""
    return DDPM(
        T_diff     = 100,
        sde_t0     = 0.4,
        n_infer    = 20,
        channels   = 64,
        depth      = 4,
        lr         = 2e-4,
        batch_size = 256,
    )


def _log_results(results: dict, description: str, git_hash: str = "xxxxxxx"):
    tsv = CACHE_DIR.parent.parent / "results.tsv"   # project root
    # fall back to CWD
    import pathlib
    for candidate in [pathlib.Path("results.tsv"),
                      CACHE_DIR / ".." / ".." / "results.tsv"]:
        if candidate.exists():
            tsv = candidate
            break
    else:
        tsv = pathlib.Path("results.tsv")

    row = "\t".join([
        git_hash[:7],
        f"{results['primary']:.6f}",
        f"{results['raw_baseline']:.6f}",
        f"{results['improvement_pct']:.2f}",
        f"{results['smoothness']:.4f}",
        f"{results['latency_ms']:.1f}",
        "keep" if results["all_guardrails_pass"] else "discard",
        description,
    ])
    with open(tsv, "a") as f:
        f.write(row + "\n")
    print(f"Logged → {tsv}")


if __name__ == "__main__":
    # ---- load data ----------------------------------------------------------
    trajs = get_trajectories()
    train_t, val_t, test_t = split_trajectories(trajs)

    # ---- load frozen decoder ------------------------------------------------
    decoder = load_or_train_decoder(train_t)

    # ---- build & train denoiser ---------------------------------------------
    denoiser = build_denoiser()
    print(f"Training denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
    denoiser.train(train_t, time_budget=TIME_BUDGET)

    # ---- evaluate -----------------------------------------------------------
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

    # ---- also check execution regime (specificity) --------------------------
    print("\nEvaluating on test set (execution regime) ...")
    exec_results = evaluate(decoder, denoiser, test_t, snr="execution")
    print(f"execution primary_rmse: {exec_results['primary']:.6f}  "
          f"(improvement: {exec_results['improvement_pct']:.2f}%)")
    print("---")

    # ---- log ----------------------------------------------------------------
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "xxxxxxx"

    desc = "unconditional DDPM baseline (SDEdit, 20 DDIM steps)"
    _log_results(results, desc, git_hash)
