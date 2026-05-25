"""
Experiment r31: Mamba-inspired selective SSM denoiser on real ECoG.

Same MambaBlock architecture as synthetic exp 31, applied to (decoded, true_DG)
pairs from BCI IV Dataset 4. Tests whether input-selective state transitions
improve over the CNN on real finger flexion data.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare_bciiv4 import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET,
                              load_dataset, split_dataset,
                              load_or_train_decoder, evaluate)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_in = expand * d_model
        self.d_in    = d_in
        self.d_state = d_state

        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, 2 * d_in, bias=False)
        self.conv1d   = nn.Conv1d(d_in, d_in, d_conv,
                                   padding=d_conv - 1, groups=d_in, bias=True)
        self.x_proj   = nn.Linear(d_in, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, d_in, bias=True)
        nn.init.constant_(self.dt_proj.bias, -4.0)

        A_init = torch.arange(1, d_state + 1, dtype=torch.float)
        A_init = A_init.unsqueeze(0).expand(d_in, -1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D     = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def forward(self, u):
        B, T, _ = u.shape
        u_norm = self.norm(u)

        xz = self.in_proj(u_norm)
        x, z = xz.split(self.d_in, dim=-1)

        x = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x = F.silu(x)

        xbc = self.x_proj(x)
        dt_raw = xbc[:, :, :1]
        B_in   = xbc[:, :, 1:1 + self.d_state]
        C_in   = xbc[:, :, 1 + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_raw))
        A  = -torch.exp(self.A_log.float())

        dA = torch.exp(dt.unsqueeze(-1) * A[None, None])
        dB = dt.unsqueeze(-1) * B_in.unsqueeze(2)

        h  = x.new_zeros(B, self.d_in, self.d_state)
        ys = []
        for t in range(T):
            h  = dA[:, t] * h + dB[:, t] * x[:, t, :, None]
            yt = (h * C_in[:, t, None, :]).sum(-1) + self.D * x[:, t]
            ys.append(yt)

        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return u + self.out_proj(y)


class DenoiseMamba(nn.Module):
    def __init__(self, dim=TRAJ_DIM, seq_len=SEQ_LEN, d_model=64,
                 n_layers=4, d_state=16):
        super().__init__()
        self.proj_in  = nn.Linear(dim, d_model)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.blocks   = nn.ModuleList(
            [MambaBlock(d_model, d_state) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, x):
        h = self.proj_in(x) + self.pos_emb(self.positions)
        for blk in self.blocks:
            h = blk(h)
        return x + self.proj_out(self.norm_out(h))


class SupervisedDenoiser:
    def __init__(self, d_model=64, n_layers=4, d_state=16,
                 lr=1e-3, batch_size=64, device=None):
        self.device  = device or ("mps"  if torch.backends.mps.is_available() else
                                   "cuda" if torch.cuda.is_available() else "cpu")
        self.model   = DenoiseMamba(d_model=d_model, n_layers=n_layers,
                                     d_state=d_state).to(self.device)
        self.opt     = torch.optim.Adam(self.model.parameters(), lr=lr,
                                         weight_decay=1e-4)
        self._mean   = self._std = None
        self.trained = False
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
    return SupervisedDenoiser(d_model=64, n_layers=4, d_state=16)


if __name__ == "__main__":
    desc = "real Mamba SSM d=64 d_state=16 n_layers=4 ECoG-HG"

    decoded, true_t = load_dataset()
    train_t, val_t, test_t, train_n, val_n, test_n = split_dataset(decoded, true_t)
    print(f"Dataset: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test")

    decoder  = load_or_train_decoder(train_t, train_n)
    denoiser = build_denoiser()

    print(f"Training Mamba denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
