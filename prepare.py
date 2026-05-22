"""
Frozen pipeline for motor imagery autoresearch.
DO NOT MODIFY — the only editable surface is denoiser.py.

Contains:
  - Trajectory generation / GRAB data loading
  - trajectory_to_neural(): population vector forward model
  - MLPDecoder: trained on execution-regime data, then frozen
  - evaluate(): primary metric + guardrails

Usage (one-time setup):
    uv run prepare.py
    uv run prepare.py --grab-dir /path/to/grab/wrist_trajectories
    uv run prepare.py --retrain   # force retrain MLP decoder
"""

import os
import sys
import math
import time
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

N_NEURONS    = 100      # simulated motor cortex neurons
TRAJ_DIM     = 3        # XYZ wrist position
SEQ_LEN      = 100      # timesteps per trajectory clip
SAMPLE_RATE  = 100      # Hz — 100 steps = 1 second of movement
TIME_BUDGET  = 600      # denoiser training budget in seconds (10 min)

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# remaining 10% is test

FIXED_SEED   = 42

CACHE_DIR    = Path(os.path.expanduser("~")) / ".cache" / "motor-imagery-autoresearch"
DECODER_PATH = CACHE_DIR / "mlp_decoder.pt"
TRAJ_PATH    = CACHE_DIR / "trajectories.npy"

# Workspace bounds (meters, centred at origin) — trajectories must stay inside
WORKSPACE_BOUNDS = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])  # (3, 2)

# Latency guardrail for real-time use
MAX_LATENCY_MS = 50.0

# ---------------------------------------------------------------------------
# Population vector model constants (fixed random seed, not editable)
# ---------------------------------------------------------------------------

_pref_rng = np.random.RandomState(FIXED_SEED)
_raw = _pref_rng.randn(N_NEURONS, TRAJ_DIM)
PREFERRED_DIRS = _raw / np.linalg.norm(_raw, axis=1, keepdims=True)  # (N_NEURONS, 3)

BASELINE_RATE = 5.0
GAIN          = 20.0

NOISE_SCALE  = {"execution": 0.2, "imagery": 0.8}
DISTORT_SCALE = {"execution": 0.0, "imagery": 0.15}

# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def _make_one_trajectory(T: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Smooth synthetic wrist trajectory via sum of sinusoids.
    Returns (T, 3) float32 in metres, centred near origin.
    """
    t = np.linspace(0, 2 * np.pi, T)
    traj = np.zeros((T, 3), dtype=np.float64)

    n_harmonics = rng.randint(3, 7)
    for _ in range(n_harmonics):
        freq  = rng.uniform(0.3, 4.0)
        amp   = rng.uniform(0.03, 0.15) / n_harmonics
        phase = rng.uniform(0, 2 * np.pi)
        dim   = rng.randint(3)
        traj[:, dim] += amp * np.sin(freq * t + phase)

    # slow drift to simulate arm reaching across workspace
    for d in range(3):
        traj[:, d] += rng.uniform(-0.08, 0.08) * (t / (2 * np.pi))

    # centre so movement starts near origin
    traj -= traj[0]
    return traj.astype(np.float32)


def _load_grab(grab_dir: Path) -> list:
    """
    Load extracted GRAB wrist trajectories from .npy files.
    Expects each file to be (T, 3) XYZ in metres.
    Segments into SEQ_LEN clips with 50 % overlap.
    """
    clips = []
    for f in sorted(grab_dir.glob("*.npy")):
        data = np.load(f)
        if data.ndim != 2 or data.shape[1] != 3:
            continue
        T = data.shape[0]
        step = SEQ_LEN // 2
        for start in range(0, T - SEQ_LEN, step):
            clip = data[start : start + SEQ_LEN].astype(np.float32)
            clips.append(clip)
    return clips


def get_trajectories(grab_dir=None, n_synthetic: int = 2000,
                     force_regenerate: bool = False) -> list:
    """
    Return list of (SEQ_LEN, 3) trajectory arrays.
    Loads GRAB if available, otherwise generates synthetic data.
    Caches to CACHE_DIR so subsequent calls are fast.
    """
    if not force_regenerate and TRAJ_PATH.exists():
        arr = np.load(TRAJ_PATH, allow_pickle=True)
        # handle both 3-D stacked arrays and legacy object arrays
        if arr.dtype == object:
            trajs = [a.astype(np.float32) for a in arr]
        else:
            trajs = [arr[i].astype(np.float32) for i in range(len(arr))]
        print(f"Trajectories: loaded {len(trajs)} clips from cache ({TRAJ_PATH})")
        return trajs

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if grab_dir is not None and Path(grab_dir).exists():
        trajs = _load_grab(Path(grab_dir))
        print(f"Trajectories: loaded {len(trajs)} clips from GRAB ({grab_dir})")
    else:
        rng = np.random.RandomState(FIXED_SEED)
        trajs = [_make_one_trajectory(SEQ_LEN, rng) for _ in range(n_synthetic)]
        print(f"Trajectories: generated {len(trajs)} synthetic clips (GRAB not found)")

    np.save(TRAJ_PATH, np.stack(trajs).astype(np.float32))   # (N, T, D)
    return trajs


def split_trajectories(trajs: list):
    """Fixed-seed train / val / test split. Returns (train, val, test)."""
    rng = np.random.RandomState(FIXED_SEED)
    idx = rng.permutation(len(trajs))
    n_train = int(len(trajs) * TRAIN_RATIO)
    n_val   = int(len(trajs) * VAL_RATIO)
    return (
        [trajs[i] for i in idx[:n_train]],
        [trajs[i] for i in idx[n_train : n_train + n_val]],
        [trajs[i] for i in idx[n_train + n_val :]],
    )


# ---------------------------------------------------------------------------
# Forward model — population vector (frozen, not editable)
# ---------------------------------------------------------------------------

def trajectory_to_neural(
    trajectory: np.ndarray,
    n_neurons: int = N_NEURONS,
    snr: str = "execution",
    seed: int = None,
) -> np.ndarray:
    """
    Biologically grounded population vector model (Georgopoulos 1986).

    trajectory : (T, 3) wrist XYZ
    returns    : (T-1, n_neurons) simulated spike counts

    Execution:  low noise, no distortion
    Imagery:    3–5× higher noise + independent distortion term
                (imagined movements have different velocity profiles)
    """
    rng = np.random.RandomState(seed)

    velocity = np.diff(trajectory, axis=0)        # (T-1, 3)

    # directional tuning: r_i(t) = baseline + gain * (v(t) · pref_i)
    rates = BASELINE_RATE + GAIN * (velocity @ PREFERRED_DIRS[:n_neurons].T)
    rates = np.clip(rates, 0.0, None)             # (T-1, n_neurons)

    noise   = NOISE_SCALE[snr]
    distort = DISTORT_SCALE[snr]

    noisy = rates + rng.randn(*rates.shape) * noise
    if distort > 0.0:
        noisy += rng.randn(*rates.shape) * distort

    return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# MLP Decoder — frozen after initial training
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    """
    Single-timestep neural-to-velocity decoder.
    Input:  (n_neurons,)  — population activity at one timestep
    Output: (3,)          — decoded wrist velocity
    """

    def __init__(self, n_neurons: int = N_NEURONS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_neurons, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, TRAJ_DIM),
        )

    def forward(self, x):
        return self.net(x)


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_mlp_decoder(
    train_trajectories: list,
    n_epochs: int = 120,
    lr: float = 1e-3,
    batch_size: int = 512,
) -> dict:
    """
    Build (neural_signal, velocity) pairs from execution-regime data,
    train MLPDecoder, save to CACHE_DIR, return checkpoint dict.
    """
    device = _get_device()
    rng = np.random.RandomState(FIXED_SEED)

    all_X, all_y = [], []
    for traj in train_trajectories:
        neural = trajectory_to_neural(traj, snr="execution",
                                      seed=int(rng.randint(1_000_000)))
        vel    = np.diff(traj, axis=0)              # (T-1, 3)
        all_X.append(neural)
        all_y.append(vel)

    X = np.concatenate(all_X, axis=0)  # (N, n_neurons)
    y = np.concatenate(all_y, axis=0)  # (N, 3)

    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std

    Xt = torch.from_numpy(Xn).float().to(device)
    yt = torch.from_numpy(yn).float().to(device)
    N  = len(Xt)

    model = MLPDecoder().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training MLP decoder: {N:,} samples × {n_epochs} epochs on {device} ...")
    for ep in range(n_epochs):
        perm  = torch.randperm(N, device=device)
        total = 0.0
        for i in range(0, N, batch_size):
            idx  = perm[i : i + batch_size]
            loss = F.mse_loss(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (ep + 1) % 30 == 0:
            print(f"  epoch {ep+1:3d}/{n_epochs}: mse={total / math.ceil(N / batch_size):.5f}")

    model.eval().cpu()
    ckpt = {
        "model_state": model.state_dict(),
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, DECODER_PATH)
    print(f"MLP decoder saved → {DECODER_PATH}")
    return ckpt


class WrappedDecoder:
    """
    Applies normalisation + MLPDecoder + velocity integration.
    Call signature:  decoded_traj = decoder(neural)
      neural       : (T-1, n_neurons)
      decoded_traj : (T, 3) positions starting at origin
    """

    def __init__(self, ckpt: dict):
        self.model  = MLPDecoder()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.X_mean = ckpt["X_mean"]
        self.X_std  = ckpt["X_std"]
        self.y_mean = ckpt["y_mean"]
        self.y_std  = ckpt["y_std"]

    def __call__(self, neural: np.ndarray) -> np.ndarray:
        Xn = (neural - self.X_mean) / self.X_std
        with torch.no_grad():
            vel_n = self.model(torch.from_numpy(Xn).float()).numpy()
        vel = vel_n * self.y_std + self.y_mean      # (T-1, 3) decoded velocities

        # integrate velocities → positions, starting at origin
        pos = np.zeros((len(vel) + 1, TRAJ_DIM), dtype=np.float32)
        pos[1:] = np.cumsum(vel, axis=0)
        return pos


def load_or_train_decoder(train_trajectories: list) -> WrappedDecoder:
    if DECODER_PATH.exists():
        ckpt = torch.load(DECODER_PATH, map_location="cpu", weights_only=False)
        print(f"MLP decoder loaded from {DECODER_PATH}")
    else:
        ckpt = train_mlp_decoder(train_trajectories)
    return WrappedDecoder(ckpt)


# ---------------------------------------------------------------------------
# Evaluation — frozen (do not modify)
# ---------------------------------------------------------------------------

def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def trajectory_smoothness(trajs: list) -> float:
    """
    Jerk-based smoothness score.  Returns mean exp(-RMS_jerk / 0.1).
    Score of 1.0 = perfectly smooth; near 0 = very jerky.
    """
    scores = []
    for t in trajs:
        if len(t) < 4:
            continue
        jerk = np.diff(t, n=3, axis=0)
        scores.append(float(np.exp(-np.sqrt(np.mean(jerk ** 2)) / 0.1)))
    return float(np.mean(scores)) if scores else 0.0


def measure_latency(denoiser, n_trials: int = 60) -> float:
    """Average per-call latency in milliseconds over n_trials."""
    dummy = np.random.randn(SEQ_LEN, TRAJ_DIM).astype(np.float32)
    for _ in range(5):           # warmup
        denoiser(dummy)
    t0 = time.perf_counter()
    for _ in range(n_trials):
        denoiser(dummy)
    return (time.perf_counter() - t0) / n_trials * 1000.0


def within_workspace(traj: np.ndarray) -> bool:
    """Return True if every point of (T, 3) traj lies within WORKSPACE_BOUNDS."""
    for d in range(TRAJ_DIM):
        if traj[:, d].min() < WORKSPACE_BOUNDS[d, 0]:
            return False
        if traj[:, d].max() > WORKSPACE_BOUNDS[d, 1]:
            return False
    return True


def _gaussian_rmse(noisy_decoded: np.ndarray, true_traj: np.ndarray,
                   sigma: float = 2.0) -> float:
    smoothed = gaussian_filter1d(noisy_decoded, sigma=sigma, axis=0)
    return rmse(smoothed, true_traj)


def is_smoothed_version(denoiser, test_trajs: list,
                        decoder: "WrappedDecoder", rtol: float = 0.05) -> bool:
    """
    Trivial guardrail: returns True if denoiser performs no better than
    a Gaussian low-pass filter (within rtol relative tolerance).
    """
    rng = np.random.RandomState(FIXED_SEED + 999)
    d_errs, g_errs = [], []

    for traj in test_trajs[:30]:
        neural   = trajectory_to_neural(traj, snr="imagery",
                                        seed=int(rng.randint(1_000_000)))
        decoded  = decoder(neural)
        decoded += traj[0:1] - decoded[0:1]   # align start

        d_errs.append(rmse(denoiser(decoded), traj))
        g_errs.append(_gaussian_rmse(decoded, traj))

    d_mean = np.mean(d_errs)
    g_mean = np.mean(g_errs)
    return abs(d_mean - g_mean) / (g_mean + 1e-8) < rtol


def evaluate(
    decoder: WrappedDecoder,
    denoiser,
    test_trajectories: list,
    snr: str = "imagery",
) -> dict:
    """
    Primary evaluation.  Returns dict with:
      primary           — mean imagery-regime RMSE after denoising (MINIMISE)
      raw_baseline      — mean RMSE before denoising
      improvement_pct   — % reduction in RMSE
      smoothness        — jerk-based score (guardrail: > 0.85)
      latency_ms        — per-call inference time (guardrail: < 50 ms)
      guardrails        — dict of bool per guardrail
      all_guardrails_pass
    """
    rng = np.random.RandomState(FIXED_SEED)
    raw_errs, den_errs, den_trajs = [], [], []

    for traj in test_trajectories:
        neural  = trajectory_to_neural(traj, snr=snr,
                                       seed=int(rng.randint(1_000_000)))
        decoded = decoder(neural)
        decoded += traj[0:1] - decoded[0:1]     # align to true start position

        denoised = denoiser(decoded)

        raw_errs.append(rmse(decoded, traj))
        den_errs.append(rmse(denoised, traj))
        den_trajs.append(denoised)

    primary      = float(np.mean(den_errs))
    raw_baseline = float(np.mean(raw_errs))
    improvement  = (raw_baseline - primary) / raw_baseline * 100.0

    smoothness  = trajectory_smoothness(den_trajs)
    latency_ms  = measure_latency(denoiser)
    trivial     = is_smoothed_version(denoiser, test_trajectories, decoder)

    guardrails = {
        "denoised_beats_raw":    primary < raw_baseline,
        "no_workspace_violation": all(within_workspace(t) for t in den_trajs),
        "smoothness":            smoothness > 0.85,
        "latency_ms":            latency_ms < MAX_LATENCY_MS,
        "not_trivial":           not trivial,
    }

    return {
        "primary":             primary,
        "raw_baseline":        raw_baseline,
        "improvement_pct":     improvement,
        "smoothness":          smoothness,
        "latency_ms":          latency_ms,
        "guardrails":          guardrails,
        "all_guardrails_pass": all(guardrails.values()),
    }


# ---------------------------------------------------------------------------
# Scaling ablation helpers
# ---------------------------------------------------------------------------

def sample_trajectory_data(hours: float, trajs: list) -> list:
    """
    Sample `hours` hours of trajectory data from `trajs`.
    Clip duration = SEQ_LEN / SAMPLE_RATE seconds.
    """
    clip_s  = SEQ_LEN / SAMPLE_RATE
    n_clips = int(hours * 3600 / clip_s)
    rng     = np.random.RandomState(FIXED_SEED)
    replace = n_clips > len(trajs)
    idx     = rng.choice(len(trajs), size=min(n_clips, len(trajs) * (3 if replace else 1)),
                         replace=replace)
    return [trajs[i] for i in idx[:n_clips]]


# ---------------------------------------------------------------------------
# Main — one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time setup: generate/verify trajectories and train MLP decoder."
    )
    parser.add_argument("--grab-dir",    default=None,
                        help="Path to extracted GRAB wrist trajectory .npy files")
    parser.add_argument("--n-synthetic", type=int, default=2000,
                        help="Synthetic trajectories to generate if GRAB unavailable")
    parser.add_argument("--retrain",     action="store_true",
                        help="Force retrain MLP decoder even if cache exists")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}\n")

    # 1. Trajectories
    trajs = get_trajectories(grab_dir=args.grab_dir, n_synthetic=args.n_synthetic)
    train_t, val_t, test_t = split_trajectories(trajs)
    print(f"Split: {len(train_t)} train / {len(val_t)} val / {len(test_t)} test\n")

    # 2. MLP decoder
    if args.retrain and DECODER_PATH.exists():
        DECODER_PATH.unlink()
    decoder = load_or_train_decoder(train_t)
    print()

    # 3. Execution–imagery gap verification (key sanity check before AutoResearch)
    print("Verifying execution–imagery gap ...")
    rng_check = np.random.RandomState(FIXED_SEED + 1)
    exec_errs, img_errs = [], []
    for traj in test_t[:60]:
        for snr_label, errs in [("execution", exec_errs), ("imagery", img_errs)]:
            neural  = trajectory_to_neural(traj, snr=snr_label,
                                           seed=int(rng_check.randint(1_000_000)))
            decoded = decoder(neural)
            decoded += traj[0:1] - decoded[0:1]
            errs.append(rmse(decoded, traj))

    exec_mean = np.mean(exec_errs)
    img_mean  = np.mean(img_errs)
    gap_pct   = (img_mean - exec_mean) / exec_mean * 100.0
    print(f"  Execution RMSE : {exec_mean:.4f} ± {np.std(exec_errs):.4f}")
    print(f"  Imagery   RMSE : {img_mean:.4f} ± {np.std(img_errs):.4f}")
    print(f"  Gap            : {gap_pct:+.1f}%")

    if gap_pct < 10:
        print("  WARNING: gap < 10 %.  Denoiser has limited headroom — consider higher imagery noise.")
    elif gap_pct >= 20:
        print("  Gap is meaningful (≥ 20 %).  AutoResearch has room to add value.")

    print("\nSetup complete.  Run 'uv run denoiser.py' to start an experiment.")
