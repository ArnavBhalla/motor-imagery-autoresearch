"""
Experiment 29: Transformer denoiser with smoothness regularization.

Same architecture as exp 28 but adds a temporal smoothness loss term:
  loss = mse(pred, target) + lambda_s * mse(diff(pred), diff(target))
where diff is the first temporal difference. This penalises jerky corrections
while still learning to project toward the clean manifold.

Exp 28 got 10.25% improvement but smoothness=0.8498 (threshold >0.85).
This variant targets smoothness compliance without sacrificing improvement.
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(h))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DenoiseTransformer(nn.Module):
    def __init__(self, dim=TRAJ_DIM, seq_len=SEQ_LEN, d_model=64, n_heads=4,
                 n_layers=4, ffn_dim=256):
        super().__init__()
        self.proj_in  = nn.Linear(dim, d_model)
        self.pos_emb  = nn.Embedding(seq_len, d_model)
        self.blocks   = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_dim) for _ in range(n_layers)]
        )
        self.proj_out = nn.Linear(d_model, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, x):
        h = self.proj_in(x) + self.pos_emb(self.positions)
        for blk in self.blocks:
            h = blk(h)
        return x + self.proj_out(h)


class SupervisedDenoiser:
    def __init__(self, n_aug=4, d_model=64, n_heads=4, n_layers=4,
                 lr=1e-3, batch_size=128, lambda_smooth=0.5, device=None):
        self.n_aug = n_aug
        self.lambda_smooth = lambda_smooth
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenoiseTransformer(d_model=d_model, n_heads=n_heads,
                                        n_layers=n_layers).to(self.device)
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
                X_list.append(decoder(neural))
                y_list.append(traj)
        return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)

    def train(self, trajectories, time_budget=TIME_BUDGET, verbose=True):
        t0 = time.perf_counter()
        decoder = load_or_train_decoder(trajectories)
        if verbose:
            print(f"  Generating {len(trajectories)}×{self.n_aug} pairs ...")
        X, y = self._make_pairs(trajectories, decoder)
        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X_n = (X - self._mean) / self._std
        y_n = (y - self._mean) / self._std
        Xt = torch.from_numpy(X_n).to(self.device)
        yt = torch.from_numpy(y_n).to(self.device)
        N = len(Xt)
        self.model.train()
        step = epoch = 0; total_loss = 0.0
        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget: break
                xb, yb = Xt[perm[i:i+self.batch_size]], yt[perm[i:i+self.batch_size]]
                pred = self.model(xb)
                mse  = F.mse_loss(pred, yb)
                # Smoothness: penalise diff in temporal derivatives
                smooth = F.mse_loss(pred[:, 1:] - pred[:, :-1],
                                    yb[:, 1:]   - yb[:, :-1])
                loss = mse + self.lambda_smooth * smooth
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
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(xt)
        return (out.squeeze(0).cpu().numpy() * self._std[0]) + self._mean[0]


def build_denoiser() -> SupervisedDenoiser:
    return SupervisedDenoiser(n_aug=4, d_model=64, n_heads=4, n_layers=4, lambda_smooth=0.5)


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
    print(f"Training smooth transformer for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "transformer smooth lambda_s=0.5 d=64 heads=4 layers=4 n_aug=4", git_hash)
