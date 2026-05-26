"""
Fast GRAB wrist extraction using rhand/lhand['params']['transl'] directly.
Resamples from 120Hz to 100Hz and saves (T, 3) float32 .npy files.
"""
import numpy as np
from pathlib import Path
from scipy.signal import resample

GRAB_DIR = Path("grab")
OUT_DIR  = Path("grab_npy")
SRC_HZ   = 120.0
DST_HZ   = 100.0

OUT_DIR.mkdir(exist_ok=True)

npz_files = sorted(GRAB_DIR.rglob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

written = 0
skipped = 0
for path in npz_files:
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        skipped += 1
        continue

    stem = path.stem
    for hand, label in [("rhand", "rw"), ("lhand", "lw")]:
        try:
            rh = data[hand].item()
            params = rh["params"]
            if hasattr(params, "item"):
                params = params.item()
            traj = params["transl"].astype(np.float32)  # (T, 3)
        except Exception:
            skipped += 1
            continue

        if traj.ndim != 2 or traj.shape[1] != 3 or traj.shape[0] < 10:
            skipped += 1
            continue

        # Resample 120Hz → 100Hz
        T_out = int(round(traj.shape[0] * DST_HZ / SRC_HZ))
        traj_r = resample(traj, T_out, axis=0).astype(np.float32)

        out_path = OUT_DIR / f"{stem}_{label}.npy"
        np.save(out_path, traj_r)
        written += 1

print(f"Done. Wrote {written} files to {OUT_DIR}/  (skipped {skipped})")
