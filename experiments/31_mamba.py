"""
Experiment 31: Mamba-inspired selective SSM denoiser (synthetic).

Implements a minimal Mamba block in pure PyTorch (no CUDA kernels required).
Key innovation over the Transformer: input-dependent state-space transitions
(selective scan) rather than global attention. Linear O(T) complexity vs O(T²).

Architecture:
  Linear embed → N × MambaBlock → Linear out (residual)

MambaBlock:
  LayerNorm → in_proj (splits x, z) → depthwise conv → SiLU →
  input-dependent (Δ, B, C) → discretize A → sequential scan → gate with z →
  out_proj

Why this might beat the Transformer:
  - Selective gating learns which timesteps matter for correction
  - Recurrent formulation gives explicit temporal memory (not just attention)
  - No quadratic attention cost (though T=100 is small, inductive bias differs)
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


class MambaBlock(nn.Module):
    """
    Pure-PyTorch Mamba block.
    Input: (B, T, d_model) — Output: (B, T, d_model) residual.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_in = expand * d_model
        self.d_in    = d_in
        self.d_state = d_state

        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_in, bias=False)
        self.conv1d   = nn.Conv1d(d_in, d_in, d_conv,
                                   padding=d_conv - 1, groups=d_in, bias=True)
        # Projects x → (dt_raw, B_in, C_in)
        self.x_proj   = nn.Linear(d_in, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_in, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)  # init Δ small → slow dynamics

        A_init = torch.arange(1, d_state + 1, dtype=torch.float)
        A_init = A_init.unsqueeze(0).expand(d_in, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D     = nn.Parameter(torch.ones(d_in))

        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def forward(self, u):
        # u: (B, T, d_model)
        B, T, _ = u.shape
        u_norm = self.norm(u)

        xz = self.in_proj(u_norm)                        # (B, T, 2*d_in)
        x, z = xz.split(self.d_in, dim=-1)

        # Causal depthwise conv (local context)
        x = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = F.silu(x)                                    # (B, T, d_in)

        # Input-dependent SSM parameters
        xbc = self.x_proj(x)                             # (B, T, d_state*2+1)
        dt_raw = xbc[:, :, :1]
        B_in   = xbc[:, :, 1:1 + self.d_state]          # (B, T, d_state)
        C_in   = xbc[:, :, 1 + self.d_state:]            # (B, T, d_state)

        dt = F.softplus(self.dt_proj(dt_raw))            # (B, T, d_in)
        A  = -torch.exp(self.A_log.float())              # (d_in, d_state)

        # Discretize: dA (B,T,d_in,d_state), dB (B,T,d_in,d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A[None, None])
        dB = dt.unsqueeze(-1) * B_in.unsqueeze(2)

        # Sequential selective scan  (T=100, fast enough without custom kernels)
        h  = x.new_zeros(B, self.d_in, self.d_state)
        ys = []
        for t in range(T):
            h  = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            yt = (h * C_in[:, t, None, :]).sum(-1) + self.D * x[:, t]
            ys.append(yt)

        y = torch.stack(ys, dim=1)                       # (B, T, d_in)
        y = y * F.silu(z)
        return u + self.out_proj(y)                      # residual


class DenoiseMamba(nn.Module):
    def __init__(self, dim=TRAJ_DIM, seq_len=SEQ_LEN, d_model=64,
                 n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj_in  = nn.Linear(dim, d_model)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.blocks   = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )
        self.norm_out  = nn.LayerNorm(d_model)
        self.proj_out  = nn.Linear(d_model, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, x):                                # (B, T, dim)
        h = self.proj_in(x) + self.pos_emb(self.positions)
        for blk in self.blocks:
            h = blk(h)
        return x + self.proj_out(self.norm_out(h))


class SupervisedDenoiser:
    def __init__(self, n_aug=4, d_model=64, n_layers=4, d_state=16,
                 lr=1e-3, batch_size=128, device=None):
        self.n_aug   = n_aug
        self.device  = device or ("mps"  if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else "cpu")
        self.model   = DenoiseMamba(d_model=d_model, n_layers=n_layers,
                                     d_state=d_state).to(self.device)
        self.opt     = torch.optim.Adam(self.model.parameters(), lr=lr)
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

        Xt = torch.from_numpy(X_n).to(self.device)
        yt = torch.from_numpy(y_n).to(self.device)
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
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(xt)
        return (out.squeeze(0).cpu().numpy() * self._std[0]) + self._mean[0]


def build_denoiser():
    return SupervisedDenoiser(n_aug=4, d_model=64, n_layers=4, d_state=16)


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
    print(f"Training Mamba denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "Mamba SSM d=64 d_state=16 n_layers=4 n_aug=4", git_hash)
