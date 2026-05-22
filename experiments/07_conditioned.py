"""
Experiment 07: Trajectory-conditioned denoiser.

Hypothesis: the decoded trajectory itself (even though noisy) contains
useful shape information about the intended movement. Conditioning the
denoiser on the observed trajectory (concatenated as extra input channels)
lets the model make targeted corrections rather than unconditional manifold
projection. This is especially useful when the decoded trajectory is
only moderately noisy (close to the manifold in some dimensions).

Architecture: input = concat(x_t, decoded_traj) → 6 channels instead of 3.
At inference: x_t is the partially-corrupted version; decoded_traj is the
original noisy decoder output (frozen as context).
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


class ConditionedDenoiseNet(nn.Module):
    """
    Takes [x_t | cond] (6 channels) as input; outputs predicted noise (3 channels).
    cond = corrupted decoded trajectory (the context).
    """
    def __init__(self, traj_dim=TRAJ_DIM, channels=64, depth=4, t_dim=32):
        super().__init__()
        self.t_emb   = SinusoidalPositionEmbedding(t_dim)
        self.t_mlp   = nn.Sequential(nn.Linear(t_dim, t_dim*2), nn.SiLU(), nn.Linear(t_dim*2, t_dim))
        self.proj_in = nn.Conv1d(traj_dim * 2, channels, 1)   # 6 input channels
        self.blocks  = nn.ModuleList([ResidualBlock1D(channels, t_dim) for _ in range(depth)])
        self.proj_out = nn.Conv1d(channels, traj_dim, 1)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, cond, t):
        """x, cond: (B, T, D) → output: (B, T, D)"""
        h = self.proj_in(torch.cat([x, cond], dim=-1).permute(0,2,1))
        t_emb = self.t_mlp(self.t_emb(t))
        for blk in self.blocks: h = blk(h, t_emb)
        return self.proj_out(h).permute(0,2,1)


def make_linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    ab    = torch.cumprod(1.0 - betas, dim=0)
    return {"alpha_bar": ab, "sqrt_ab": ab.sqrt(), "sqrt_1mab": (1-ab).sqrt()}


class ConditionedDDPM:
    """
    Conditioned DDPM: at training, condition on a noisy version of x0.
    At inference, condition on the decoded trajectory.
    """
    def __init__(self, T_diff=100, sde_t0=0.5, n_infer=20,
                 channels=64, depth=4, lr=2e-4, batch_size=256, t_dim=32,
                 cond_noise=0.1, device=None):
        self.T_diff = T_diff; self.sde_t0 = sde_t0; self.n_infer = n_infer
        self.cond_noise = cond_noise   # noise added to conditioning signal during training
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self.sched  = {k: v.to(self.device) for k,v in make_linear_schedule(T_diff).items()}
        self.model  = ConditionedDenoiseNet(channels=channels, depth=depth, t_dim=t_dim).to(self.device)
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
                x0    = dataset[perm[i:i+self.batch_size]]; B = len(x0)
                t_idx = torch.randint(0, self.T_diff, (B,), device=self.device)
                eps   = torch.randn_like(x0)
                x_t   = self.sched["sqrt_ab"][t_idx].view(B,1,1)*x0 + self.sched["sqrt_1mab"][t_idx].view(B,1,1)*eps
                # condition = x0 + small noise (simulates the decoded trajectory at train time)
                cond  = x0 + torch.randn_like(x0) * self.cond_noise
                loss  = F.mse_loss(self.model(x_t, cond, t_idx.float()), eps)
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
        x    = (decoded_traj - self._mean[0]) / self._std[0]
        cond = torch.from_numpy(x).float().unsqueeze(0).to(self.device)   # fixed conditioning
        x_t  = cond.clone()   # start from decoded trajectory

        t0   = int(self.sde_t0 * self.T_diff)
        x_t  = self.sched["sqrt_ab"][t0].item()*x_t + self.sched["sqrt_1mab"][t0].item()*torch.randn_like(x_t)
        ts   = torch.linspace(t0, 0, self.n_infer+1, dtype=torch.long, device=self.device).clamp(0, self.T_diff-1)
        self.model.eval()
        with torch.no_grad():
            for i in range(self.n_infer):
                t_c = ts[i].unsqueeze(0); t_p = ts[i+1].unsqueeze(0)
                ab_c = self.sched["alpha_bar"][t_c].view(1,1,1)
                ab_p = self.sched["alpha_bar"][t_p].view(1,1,1)
                pred = self.model(x_t, cond, t_c.float())
                x0h  = ((x_t - (1-ab_c).sqrt()*pred) / ab_c.sqrt()).clamp(-3,3)
                x_t  = ab_p.sqrt()*x0h + (1-ab_p).sqrt()*pred
        return x_t.squeeze(0).cpu().numpy() * self._std[0] + self._mean[0]


DDPM = ConditionedDDPM   # alias for orchestrator


def build_denoiser():
    return ConditionedDDPM(T_diff=100, sde_t0=0.5, n_infer=20,
                           channels=64, depth=4, lr=2e-4, batch_size=256,
                           cond_noise=0.3)   # cond_noise matches imagery distortion level


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
    print(f"Training conditioned denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "conditioned DDPM (decoded traj as context, cond_noise=0.3)", git_hash)
