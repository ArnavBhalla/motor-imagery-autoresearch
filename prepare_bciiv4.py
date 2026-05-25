"""
prepare_bciiv4.py — data module for BCI Competition IV Dataset 4.

ECoG channel counts vary by subject (sub1=62, sub2=48, sub3=64), so we apply
per-subject decoders inside load_dataset() and return already-decoded trajectories.
The denoiser always operates in 5-finger DataGlove space regardless of subject.

Public API:
    load_dataset()                  → (decoded_trajs, true_trajs)
    split_dataset(decoded, true)    → (train_d, val_d, test_d, train_t, val_t, test_t)
    load_or_train_decoder(d, t)     → identity callable (decoding already applied)
    evaluate(decoder, denoiser, test_decoded, test_true) → results dict

'decoded_trajs' are the per-subject CNN decoder outputs (noisy estimates).
'true_trajs'    are the actual DataGlove recordings (clean ground truth).
"""

import os
import time
import pickle

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BCIIV4_DIR  = "bciiv4"
SUBJECTS    = [
    ("sub1_comp.mat", 62),
    ("sub2_comp.mat", 48),
    ("sub3_comp.mat", 64),
]

SEQ_LEN     = 100     # timesteps per window (1 s at 100 Hz)
TRAJ_DIM    = 5       # DataGlove fingers
FS_ORIG     = 1000    # original sampling rate (Hz)
FS_NEW      = 100     # target feature rate (Hz)
BIN_SAMPLES = FS_ORIG // FS_NEW   # 10 samples per bin

TIME_BUDGET = 600     # seconds per experiment
FIXED_SEED  = 42
STRIDE      = 50      # 50% overlap → 2× windows

CACHE_DIR   = os.path.expanduser("~/.cache/motor-imagery-autoresearch/bciiv4")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _hg_power(ecog, n_ch):
    """
    Raw ECoG (N, n_ch) at 1000 Hz → high-gamma power (N//10, n_ch) at 100 Hz.
    Bandpass 70-170 Hz → square → 10ms non-overlapping mean.
    """
    b, a = sig.butter(4, [70.0, 170.0], btype="bandpass", fs=float(FS_ORIG))
    filtered = sig.filtfilt(b, a, ecog, axis=0).astype(np.float32)
    n_bins   = len(filtered) // BIN_SAMPLES
    filtered = filtered[:n_bins * BIN_SAMPLES]
    power    = (filtered ** 2).reshape(n_bins, BIN_SAMPLES, n_ch).mean(axis=1)
    return power.astype(np.float32)  # (n_bins, n_ch)


def _decimate_dg(dg):
    """Average DataGlove (N, 5) in 10ms bins → (N//10, 5) at 100 Hz."""
    n_bins = len(dg) // BIN_SAMPLES
    dg     = dg[:n_bins * BIN_SAMPLES]
    return dg.reshape(n_bins, BIN_SAMPLES, TRAJ_DIM).mean(axis=1).astype(np.float32)


def _standardize_cols(arr):
    """Zero-mean, unit-std per column (channel)."""
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-8
    return (arr - mean) / std


def _window(features, labels, seq_len=SEQ_LEN, stride=STRIDE):
    n = features.shape[0]
    fw, lw = [], []
    for start in range(0, n - seq_len + 1, stride):
        fw.append(features[start:start + seq_len])
        lw.append(labels[start:start + seq_len])
    return np.stack(fw), np.stack(lw)

# ---------------------------------------------------------------------------
# Per-subject CNN decoder
# ---------------------------------------------------------------------------

class _SubjectDecoder(nn.Module):
    """(B, n_ch, SEQ_LEN) → (B, TRAJ_DIM, SEQ_LEN) via 1-D CNN."""
    def __init__(self, n_ch, hidden=256, kernel=9):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_ch,   hidden, kernel, padding=pad), nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel, padding=pad), nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, TRAJ_DIM, kernel, padding=pad),
        )

    def forward(self, x):
        return self.net(x)


def _train_subject_decoder(X_feat, y_dg, n_ch, device,
                             lr=1e-3, epochs=300, batch_size=64):
    """Train one decoder for a single subject. Returns eval-mode model."""
    model = _SubjectDecoder(n_ch).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.from_numpy(X_feat.transpose(0, 2, 1)).to(device)  # (N, n_ch, T)
    yt = torch.from_numpy(y_dg.transpose(0, 2, 1)).to(device)    # (N, 5, T)
    N  = len(Xt)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            xb = Xt[perm[i:i + batch_size]]
            yb = yt[perm[i:i + batch_size]]
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        if (ep + 1) % 100 == 0:
            print(f"    decoder ep {ep+1:3d} loss {total:.5f}")

    model.eval()
    return model

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset():
    """
    Returns:
        decoded_trajs: list of (SEQ_LEN, TRAJ_DIM) — CNN decoder output (noisy estimates)
        true_trajs:    list of (SEQ_LEN, TRAJ_DIM) — DataGlove ground truth
    """
    cache_path = os.path.join(CACHE_DIR, "dataset_v3.pkl")
    if os.path.exists(cache_path):
        print("  Loading cached dataset ...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else "cpu")

    all_decoded, all_true = [], []

    for fname, n_ch in SUBJECTS:
        print(f"  Processing {fname} ({n_ch} channels) ...")
        path = os.path.join(BCIIV4_DIR, fname)
        mat  = sio.loadmat(path)
        ecog = mat["train_data"].astype(np.float32)   # (400000, n_ch)
        dg   = mat["train_dg"].astype(np.float32)     # (400000, 5)

        # Notch-filter powerline before HG extraction
        for freq in [60, 120, 180]:
            b, a = sig.iirnotch(freq, 35.0, float(FS_ORIG))
            ecog = sig.filtfilt(b, a, ecog, axis=0).astype(np.float32)

        features = _hg_power(ecog, n_ch)           # (T_100, n_ch)
        labels   = _decimate_dg(dg)                # (T_100, 5)
        features = _standardize_cols(features)

        n = min(len(features), len(labels))
        features, labels = features[:n], labels[:n]

        feat_wins, dg_wins = _window(features, labels)  # (W, 100, n_ch) / (W, 100, 5)
        W = len(feat_wins)

        # Train/use subject-specific decoder (80% train split within subject)
        n_train = int(0.8 * W)
        X_tr = feat_wins[:n_train]
        y_tr = dg_wins[:n_train]

        subj_cache = os.path.join(CACHE_DIR, f"decoder_{fname}.pt")
        model = _SubjectDecoder(n_ch).to(device)
        if os.path.exists(subj_cache):
            model.load_state_dict(torch.load(subj_cache, map_location=device,
                                              weights_only=True))
            model.eval()
            print(f"    Loaded cached decoder for {fname}")
        else:
            print(f"    Training decoder for {fname} ({n_train} windows) ...")
            model = _train_subject_decoder(X_tr, y_tr, n_ch, device)
            torch.save(model.state_dict(), subj_cache)

        # Decode all windows
        Xt = torch.from_numpy(feat_wins.transpose(0, 2, 1)).to(device)
        with torch.no_grad():
            decoded = model(Xt).permute(0, 2, 1).cpu().numpy()  # (W, 100, 5)

        all_decoded.extend(list(decoded))
        all_true.extend(list(dg_wins))

    result = (all_decoded, all_true)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Dataset cached: {len(all_decoded)} windows total")
    return result


def split_dataset(decoded_trajs, true_trajs, seed=FIXED_SEED):
    """Deterministic 80/10/10 split."""
    rng = np.random.RandomState(seed)
    n   = len(decoded_trajs)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    def pick(lst, idxs):
        return [lst[i] for i in idxs]

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    return (pick(decoded_trajs, train_idx),
            pick(decoded_trajs, val_idx),
            pick(decoded_trajs, test_idx),
            pick(true_trajs,    train_idx),
            pick(true_trajs,    val_idx),
            pick(true_trajs,    test_idx))


def load_or_train_decoder(decoded_trajs, true_trajs):
    """
    Per-subject decoding is already applied in load_dataset().
    Returns an identity function so experiment files stay structurally compatible.
    """
    return lambda x: x


def evaluate(decoder, denoiser, test_decoded, test_true):
    """
    decoder  — identity (pre-decoded in load_dataset)
    denoiser — the model under test
    Returns dict: primary_rmse, raw_baseline, improvement_pct,
                  smoothness, latency_ms, all_pass, guardrails.
    """
    # Baseline: decoded vs true
    raw_errs = [float(np.sqrt(((d - t) ** 2).mean()))
                for d, t in zip(test_decoded, test_true)]
    raw_baseline = float(np.mean(raw_errs))

    # Denoised
    t0 = time.perf_counter()
    denoised  = [denoiser(d) for d in test_decoded]
    latency_ms = (time.perf_counter() - t0) / len(test_decoded) * 1000.0

    den_errs = [float(np.sqrt(((d - t) ** 2).mean()))
                for d, t in zip(denoised, test_true)]
    primary  = float(np.mean(den_errs))

    improvement_pct = (raw_baseline - primary) / raw_baseline * 100.0

    def _roughness(trajs):
        return float(np.mean([np.std(np.diff(t, axis=0)) for t in trajs]))

    raw_rough  = _roughness(test_decoded)
    den_rough  = _roughness(denoised)
    smoothness = float(min(1.0, raw_rough / (den_rough + 1e-8)))

    guardrails = {
        "denoised_beats_raw": primary < raw_baseline,
        "smoothness_ok":      smoothness >= 0.85,
        "latency_ok":         latency_ms < 50.0,
    }
    all_pass = all(guardrails.values())

    return {
        "primary":         primary,
        "raw_baseline":    raw_baseline,
        "improvement_pct": improvement_pct,
        "smoothness":      smoothness,
        "latency_ms":      latency_ms,
        "guardrails":      guardrails,
        "all_pass":        all_pass,
    }
