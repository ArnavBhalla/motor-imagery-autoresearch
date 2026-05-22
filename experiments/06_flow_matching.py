"""
Experiment 06: Flow matching (Lipman et al. 2022 / Liu et al. 2022).

Hypothesis: flow matching uses straight-line ODE paths from noise → data,
leading to faster convergence and fewer function evaluations at inference
than DDPM. With fewer NFE we stay under the 50ms latency guardrail while
potentially learning a better trajectory manifold.

Key changes:
  Training: sample t ~ U[0,1], x_t = (1-t)*x0 + t*noise,
            model predicts velocity v = noise - x0.
  Inference: ODE solve from t=1 → t=sde_t0 (partial denoise) using Euler.
"""

import math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freqs",
            torch.exp(-math.log(10_000) * torch.arange(dim//2) / (dim//2 - 1)))
    def forward(self, t):
        args = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, t_dim, kernel=5):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.t_proj = nn.Linear(t_dim, channels * 2)
    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        ts = self.t_proj(t_emb).unsqueeze(-1)
        scale, shift = ts.chunk(2, dim=1)
        return x + F.silu(self.conv2(self.norm2(h) * (1+scale) + shift))


class VelocityNet(nn.Module):
    """Predicts flow velocity v(x_t, t) = noise - x0."""
    def __init__(self, traj_dim=TRAJ_DIM, channels=64, depth=4, t_dim=32):
        super().__init__()
        self.t_emb   = SinusoidalPositionEmbedding(t_dim)
        self.t_mlp   = nn.Sequential(nn.Linear(t_dim, t_dim*2), nn.SiLU(), nn.Linear(t_dim*2, t_dim))
        self.proj_in = nn.Conv1d(traj_dim, channels, 1)
        self.blocks  = nn.ModuleList([ResidualBlock1D(channels, t_dim) for _ in range(depth)])
        self.proj_out = nn.Conv1d(channels, traj_dim, 1)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
    def forward(self, x, t):
        h = self.proj_in(x.permute(0,2,1))
        t_emb = self.t_mlp(self.t_emb(t * 100))   # scale to match sinusoidal range
        for blk in self.blocks: h = blk(h, t_emb)
        return self.proj_out(h).permute(0,2,1)


class FlowMatching:
    """
    Flow matching denoiser.
    SDEdit-style inference: corrupt decoded trajectory to t=sde_t0,
    then run Euler ODE backward from t=sde_t0 → 0.
    """

    def __init__(self, sde_t0=0.6, n_infer=15,
                 channels=64, depth=4, lr=2e-4, batch_size=256, t_dim=32, device=None):
        self.sde_t0 = sde_t0; self.n_infer = n_infer
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self.model  = VelocityNet(channels=channels, depth=depth, t_dim=t_dim).to(self.device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size; self._mean = self._std = None; self.trained = False

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        arr = np.stack(trajectories).astype(np.float32)
        self._mean = arr.mean(axis=(0,1), keepdims=True)
        self._std  = arr.std(axis=(0,1),  keepdims=True) + 1e-8
        arr = (arr - self._mean) / self._std
        dataset = torch.from_numpy(arr).to(self.device); N = len(dataset)
        self.model.train(); step = epoch = 0; total_loss = 0.0; t0 = time.perf_counter()
        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget: break
                x0   = dataset[perm[i:i+self.batch_size]]; B = len(x0)
                # flow matching: interpolate between data and noise
                t    = torch.rand(B, device=self.device)                # t ~ U[0,1]
                eps  = torch.randn_like(x0)
                x_t  = (1 - t.view(B,1,1)) * x0 + t.view(B,1,1) * eps  # straight line
                v_true = eps - x0                                         # target velocity

                loss = F.mse_loss(self.model(x_t, t), v_true)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step(); total_loss += loss.item(); step += 1
            epoch += 1
            if time.perf_counter() - t0 >= time_budget: break
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | loss {total_loss/step:.5f} | elapsed {time.perf_counter()-t0:.0f}s")
        self.model.eval(); self.trained = True
        if verbose: print(f"Training done: {epoch} epochs, {step} steps, {time.perf_counter()-t0:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained: return decoded_traj
        x   = (decoded_traj - self._mean[0]) / self._std[0]
        x_t = torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        # Forward: corrupt to t=sde_t0 on the flow straight line
        t_start = self.sde_t0
        eps_0   = torch.randn_like(x_t)
        x_t     = (1 - t_start) * x_t + t_start * eps_0

        # Euler ODE backward from sde_t0 → 0
        ts  = torch.linspace(t_start, 0.0, self.n_infer + 1)
        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_infer):
                t_cur  = ts[i].item()
                dt     = ts[i] - ts[i+1]          # positive (going backward)
                t_tens = torch.tensor([t_cur], device=self.device)
                v      = self.model(x_t, t_tens)
                x_t    = x_t - dt.item() * v      # Euler step backward

        return x_t.squeeze(0).cpu().numpy() * self._std[0] + self._mean[0]


# Alias so research_loop.py can use generic DDPM handle
DDPM = FlowMatching


def build_denoiser():
    return FlowMatching(sde_t0=0.6, n_infer=15, channels=64, depth=4, lr=2e-4, batch_size=256)


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
    print(f"Training flow matching denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "flow matching sde_t0=0.6 n_infer=15 Euler", git_hash)
