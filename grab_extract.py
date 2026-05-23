"""
GRAB dataset extraction — converts GRAB .npz sequences to (T, 3) wrist .npy files.

GRAB download (requires free account):
    https://grab.is.tue.mpg.de/  →  Data  →  download each subject zip (S1–S10)
    Unzip into a single directory, e.g. ~/data/grab/
    Expected structure:
        ~/data/grab/
            s1/
                s1_apple_eat_1.npz
                s1_apple_eat_2.npz
                ...
            s2/
                ...

Usage:
    uv run grab_extract.py --grab-dir ~/data/grab --out-dir grab_npy
    uv run prepare.py --grab-dir grab_npy --retrain

How it works:
    Each GRAB .npz sequence has pre-computed joint positions in
    data['body']['joints'] shaped (T, J, 3).  SMPLX joint 21 is
    the right wrist; joint 20 is the left wrist.  We extract both,
    saving each as a separate (T, 3) .npy file in --out-dir.

    Falls back to extracting from SMPLX params via the `smplx` package
    if precomputed joints are absent (uncommon in newer GRAB releases).
"""

import argparse
import numpy as np
from pathlib import Path

# SMPLX joint indices for wrists
RIGHT_WRIST = 21
LEFT_WRIST  = 20


def _extract_from_joints(joints: np.ndarray) -> tuple:
    """Return (right_wrist, left_wrist) trajectory arrays, each (T, 3)."""
    return joints[:, RIGHT_WRIST, :], joints[:, LEFT_WRIST, :]


def _extract_from_smplx(params: dict, device: str = "cpu") -> tuple:
    """Run SMPLX forward pass to get wrist positions. Requires `smplx` package."""
    try:
        import smplx, torch
    except ImportError:
        raise RuntimeError("smplx not installed. Run: uv pip install smplx")

    T = params["transl"].shape[0]
    model = smplx.create(model_type="smplx", gender="neutral",
                         use_face_contour=False, num_betas=10,
                         num_expression_coeffs=10, use_pca=True, num_pca_comps=6)
    model = model.to(device)

    def _t(x): return torch.from_numpy(x).float().to(device)

    with torch.no_grad():
        out = model(
            transl=_t(params["transl"]),
            global_orient=_t(params["global_orient"]),
            body_pose=_t(params["body_pose"]),
            betas=_t(params.get("betas", np.zeros((T, 10)))),
            return_verts=False,
        )
    joints = out.joints.cpu().numpy()  # (T, J, 3)
    return joints[:, RIGHT_WRIST, :], joints[:, LEFT_WRIST, :]


def process_file(path: Path, out_dir: Path) -> int:
    """
    Extract wrist trajectories from one GRAB .npz file.
    Returns number of files written (0, 1, or 2).
    """
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"  SKIP {path.name}: load error — {e}")
        return 0

    # Try to get precomputed joints first (fast path)
    right_w = left_w = None
    try:
        body = data["body"].item() if data["body"].ndim == 0 else data["body"]
        if isinstance(body, dict) and "joints" in body:
            joints = body["joints"]
            if joints.ndim == 3 and joints.shape[1] >= 22:
                right_w, left_w = _extract_from_joints(joints)
        elif "joints" in data:
            joints = data["joints"]
            if joints.ndim == 3 and joints.shape[1] >= 22:
                right_w, left_w = _extract_from_joints(joints)
    except Exception:
        pass

    # Fall back to SMPLX forward pass
    if right_w is None:
        try:
            body = data["body"].item() if data["body"].ndim == 0 else data["body"]
            params = body["params"].item() if body["params"].ndim == 0 else body["params"]
            # params should have transl (T,3), global_orient (T,3), body_pose (T,63)
            if "transl" in params:
                right_w, left_w = _extract_from_smplx(params)
        except Exception as e:
            print(f"  SKIP {path.name}: could not extract joints — {e}")
            return 0

    if right_w is None:
        print(f"  SKIP {path.name}: unrecognised data structure")
        return 0

    stem = path.stem
    written = 0
    for label, traj in [("rw", right_w), ("lw", left_w)]:
        if traj is None or traj.shape[0] < 10:
            continue
        out_path = out_dir / f"{stem}_{label}.npy"
        np.save(out_path, traj.astype(np.float32))
        written += 1

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grab-dir", required=True,
                    help="Root directory of downloaded GRAB data (contains s1/, s2/, …)")
    ap.add_argument("--out-dir", default="grab_npy",
                    help="Output directory for extracted wrist .npy files")
    args = ap.parse_args()

    grab_dir = Path(args.grab_dir).expanduser()
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(grab_dir.rglob("*.npz"))
    if not npz_files:
        print(f"No .npz files found under {grab_dir}")
        print("Download GRAB data from https://grab.is.tue.mpg.de/")
        return

    print(f"Found {len(npz_files)} .npz files in {grab_dir}")
    total = 0
    for i, f in enumerate(npz_files):
        n = process_file(f, out_dir)
        total += n
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(npz_files)}] written {total} wrist trajectories so far")

    print(f"\nDone. Wrote {total} wrist trajectory files to {out_dir}/")
    print(f"\nNext steps:")
    print(f"  uv run prepare.py --grab-dir {out_dir} --retrain")
    print(f"  # Then restart the research loop.")


if __name__ == "__main__":
    main()
