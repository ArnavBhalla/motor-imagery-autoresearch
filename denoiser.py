"""
Experiment 30: JEPA-inspired trajectory denoiser.

Adapts Joint-Embedding Predictive Architecture (Assran et al., 2023) to
supervised 1D trajectory correction.

Architecture:
  - Context encoder:  noisy_decoded (T,3) → latent sequence (T, d_latent)
  - Target encoder:   clean traj (T,3)    → target latents  (T, d_latent)
                      [EMA of context encoder, stop-gradient]
  - Predictor:        (T, d_latent) noisy → (T, d_latent) predicted
  - Decoder:          (T, d_latent) → (T, 3) [auxiliary reconstruction head]

Loss:
  L = MSE(z_pred, sg(z_clean))          [JEPA latent loss]
    + lambda_rec * MSE(decode(z_pred), x_clean)  [reconstruction]
    + lambda_smooth * MSE(diff(z_pred), diff(sg(z_clean)))  [temporal smoothness]

Why over direct supervised CNN:
  - EMA target encoder provides stable, slowly-evolving latent targets
    (reduces training instability on small datasets)
  - Predicting in latent space forces encoder to learn noise-robust
    representations rather than memorising (noisy → clean) pixel mappings
  - Predictor only learns clean-manifold geometry, not decoder reconstruction
"""

import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import (SEQ_LEN, TRAJ_DIM, TIME_BUDGET, FIXED_SEED,
                     get_trajectories, split_trajectories,
                     load_or_train_decoder, evaluate,
                     trajectory_to_neural)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    """Maps (B, T, dim_in) → (B, T, d_latent) via 1-D convolutions."""
    def __init__(self, dim_in=TRAJ_DIM, d_latent=64, kernel=9):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim_in,   d_latent, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(d_latent, d_latent, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(d_latent, d_latent, kernel, padding=pad),
        )
        self.norm = nn.LayerNorm(d_latent)

    def forward(self, x):          # x: (B, T, dim_in)
        h = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, d_latent)
        return self.norm(h)


class TransformerPredictor(nn.Module):
    """Maps (B, T, d_latent) → (B, T, d_latent) using self-attention."""
    def __init__(self, d_latent=64, n_heads=4, n_layers=2, ffn_dim=128, seq_len=SEQ_LEN):
        super().__init__()
        self.pos = nn.Embedding(seq_len, d_latent)
        layer   = nn.TransformerEncoderLayer(d_latent, n_heads, ffn_dim,
                                              dropout=0.1, batch_first=True,
                                              activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.register_buffer("positions", torch.arange(seq_len))

    def forward(self, z):          # z: (B, T, d_latent)
        return self.tf(z + self.pos(self.positions))


class ConvDecoder(nn.Module):
    """Maps (B, T, d_latent) → (B, T, dim_out) via 1-D convolutions."""
    def __init__(self, d_latent=64, dim_out=TRAJ_DIM, kernel=9):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_latent, d_latent, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(d_latent, dim_out,  kernel, padding=pad),
        )

    def forward(self, z):
        return self.net(z.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Full JEPA denoiser
# ---------------------------------------------------------------------------

class JEPADenoiser:
    def __init__(self, n_aug=4, d_latent=64, ema_decay=0.99,
                 lambda_rec=1.0, lambda_smooth=0.3,
                 lr=1e-3, batch_size=128, device=None):
        self.n_aug         = n_aug
        self.ema_decay     = ema_decay
        self.lambda_rec    = lambda_rec
        self.lambda_smooth = lambda_smooth
        self.batch_size    = batch_size
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")

        self.ctx_enc  = ConvEncoder(d_latent=d_latent).to(self.device)
        self.tgt_enc  = copy.deepcopy(self.ctx_enc)   # EMA target — no grad
        for p in self.tgt_enc.parameters():
            p.requires_grad_(False)

        self.predictor = TransformerPredictor(d_latent=d_latent).to(self.device)
        self.decoder   = ConvDecoder(d_latent=d_latent).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.ctx_enc.parameters()) +
            list(self.predictor.parameters()) +
            list(self.decoder.parameters()),
            lr=lr
        )
        self._mean = self._std = None
        self.trained = False

    def _ema_update(self):
        with torch.no_grad():
            for p_t, p_c in zip(self.tgt_enc.parameters(),
                                self.ctx_enc.parameters()):
                p_t.data.mul_(self.ema_decay).add_(p_c.data, alpha=1 - self.ema_decay)

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
        mlp_decoder = load_or_train_decoder(trajectories)
        if verbose:
            print(f"  Generating {len(trajectories)}×{self.n_aug} pairs ...")
        X, y = self._make_pairs(trajectories, mlp_decoder)

        self._mean = X.mean(axis=(0, 1), keepdims=True)
        self._std  = X.std(axis=(0, 1),  keepdims=True) + 1e-8
        Xn = (X - self._mean) / self._std
        yn = (y - self._mean) / self._std

        Xt = torch.from_numpy(Xn).to(self.device)   # (N, T, 3)
        yt = torch.from_numpy(yn).to(self.device)

        N = len(Xt)
        self.ctx_enc.train(); self.predictor.train(); self.decoder.train()
        step = epoch = 0; total_loss = 0.0

        while True:
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                if time.perf_counter() - t0 >= time_budget: break
                xb = Xt[perm[i:i+self.batch_size]]   # noisy
                yb = yt[perm[i:i+self.batch_size]]   # clean

                # Context path
                z_noisy = self.ctx_enc(xb)            # (B, T, d_latent)
                z_pred  = self.predictor(z_noisy)

                # Target path — EMA encoder, stop-gradient
                with torch.no_grad():
                    z_clean = self.tgt_enc(yb)

                # Auxiliary reconstruction
                recon = self.decoder(z_pred)          # (B, T, 3)

                # Losses
                latent_loss = F.mse_loss(z_pred, z_clean)
                recon_loss  = F.mse_loss(recon, yb)
                smooth_loss = F.mse_loss(z_pred[:, 1:] - z_pred[:, :-1],
                                         z_clean[:, 1:] - z_clean[:, :-1])
                loss = latent_loss + self.lambda_rec * recon_loss + self.lambda_smooth * smooth_loss

                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.ctx_enc.parameters()) +
                    list(self.predictor.parameters()) +
                    list(self.decoder.parameters()), 1.0)
                self.opt.step()
                self._ema_update()
                total_loss += loss.item(); step += 1

            epoch += 1
            if time.perf_counter() - t0 >= time_budget: break
            if verbose and epoch % 50 == 0:
                print(f"  epoch {epoch:4d} | step {step:6d} | loss {total_loss/step:.5f} | "
                      f"elapsed {time.perf_counter()-t0:.0f}s")

        self.ctx_enc.eval(); self.predictor.eval(); self.decoder.eval()
        self.trained = True
        if verbose:
            print(f"Training done: {epoch} epochs, {step} steps, {time.perf_counter()-t0:.1f}s")

    def __call__(self, decoded_traj):
        if not self.trained: return decoded_traj
        x = (decoded_traj.astype(np.float32) - self._mean[0]) / self._std[0]
        xt = torch.from_numpy(x).float().unsqueeze(0).to(self.device)   # (1, T, 3)
        with torch.no_grad():
            z = self.ctx_enc(xt)
            z_pred = self.predictor(z)
            out = self.decoder(z_pred)
        return (out.squeeze(0).cpu().numpy() * self._std[0]) + self._mean[0]


def build_denoiser() -> JEPADenoiser:
    return JEPADenoiser(n_aug=4, d_latent=64, ema_decay=0.99,
                        lambda_rec=1.0, lambda_smooth=0.3)


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
    print(f"Training JEPA denoiser for {TIME_BUDGET}s on {denoiser.device} ...")
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
    _log_results(results, "JEPA d_latent=64 ema=0.99 lambda_rec=1.0 lambda_smooth=0.3 n_aug=4", git_hash)
