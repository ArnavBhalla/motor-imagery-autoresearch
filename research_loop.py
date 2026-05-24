"""
Autonomous overnight research loop.

Applies pre-defined experiment files from experiments/ to denoiser.py,
runs each, records results in results.tsv, keeps or discards via git.
Loops through the queue repeatedly until interrupted (Ctrl-C or kill).

Usage:
    nohup uv run research_loop.py > research.log 2>&1 &
    tail -f research.log        # to monitor progress
    kill %1                     # to stop

Results are logged to results.tsv.
"""

import os
import sys
import shutil
import subprocess
import signal
import time
import re
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Experiment queue (in priority order)
# ---------------------------------------------------------------------------
# Each entry: (experiment_file, description)
# The loop copies the file to denoiser.py, commits, runs, keeps/discards.

EXPERIMENTS_DIR = Path("experiments")

QUEUE = [
    # Transformer — global receptive field via self-attention over 100 timesteps
    ("28_transformer.py",              "transformer denoiser d=64 heads=4 layers=4 n_aug=4"),
    # Big model + augmentation — combines both winning CNN ingredients
    ("27_supervised_big_aug.py",       "supervised CNN big+aug ch=128 d=6 n_aug=4"),
    # 8x augmentation — more training diversity
    ("26_supervised_aug8.py",          "supervised CNN aug8 n_aug=8 channels=64 depth=4"),
    # Current best for stochastic exploration
    ("25_supervised_cnn_augmented.py", "supervised CNN augmented n_aug=4 channels=64 depth=4"),
]

TIMEOUT_SECONDS = 900   # 15 min hard kill per run (matches program.md)
RUN_LOG = Path("run.log")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True)


def short_hash():
    r = git("rev-parse", "--short", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else "xxxxxxx"


def parse_results(log_path: Path):
    """
    Parse primary_rmse, raw_baseline, improvement_pct, smoothness,
    latency_ms, all_pass from run.log. Returns None on failure.
    """
    try:
        text = log_path.read_text()
    except Exception:
        return None

    def find(key):
        m = re.search(rf"^{re.escape(key)}:\s+([^\n]+)", text, re.MULTILINE)
        return m.group(1).strip() if m else None

    primary      = find("primary_rmse")
    raw_baseline = find("raw_baseline")
    imprv        = find("improvement_pct")
    smooth       = find("smoothness")
    latency      = find("latency_ms")
    all_pass_str = find("all_pass")

    if primary is None:
        return None

    return {
        "primary":         float(primary),
        "raw_baseline":    float(raw_baseline) if raw_baseline else None,
        "improvement_pct": float(imprv.rstrip("%")) if imprv else None,
        "smoothness":      float(smooth)  if smooth  else None,
        "latency_ms":      float(latency) if latency else None,
        "all_pass":        all_pass_str == "True",
    }


def log_to_tsv(results: dict, description: str, status: str, git_hash: str):
    row = "\t".join([
        git_hash,
        f"{results['primary']:.6f}",
        f"{results.get('raw_baseline', 0):.6f}",
        f"{results.get('improvement_pct', 0):.2f}",
        f"{results.get('smoothness', 0):.4f}",
        f"{results.get('latency_ms', 0):.1f}",
        status,
        description,
    ])
    with open("results.tsv", "a") as f:
        f.write(row + "\n")
    print(f"  → logged ({status}): {description}")


def run_experiment(src_file: str, description: str, best_rmse: float, iteration: int):
    """
    Copy experiment file → denoiser.py, commit, run, parse, keep/discard.
    Returns updated best_rmse.
    """
    print(f"\n{'='*60}")
    print(f"[iter {iteration:03d}] {description}")
    print(f"{'='*60}")

    # Resolve source path
    if src_file == "denoiser.py":
        # baseline: use whatever denoiser.py already contains from git
        # (it's the baseline we committed initially)
        src_path = None   # no copy needed
    else:
        src_path = EXPERIMENTS_DIR / src_file

    if src_path is not None:
        if not src_path.exists():
            print(f"  SKIP: {src_path} not found")
            return best_rmse
        shutil.copy(src_path, "denoiser.py")
        print(f"  Applied: {src_path} → denoiser.py")

    # Commit
    git("add", "denoiser.py")
    commit_msg = f"[autoresearch iter {iteration:03d}] {description}"
    r = git("commit", "-m", commit_msg)
    if r.returncode != 0 and "nothing to commit" not in r.stdout + r.stderr:
        # Force commit even if file is "same" (paranoia)
        git("commit", "--allow-empty", "-m", commit_msg)
    h = short_hash()
    print(f"  Commit: {h}")

    # Run
    print(f"  Running experiment (timeout={TIMEOUT_SECONDS}s) ...")
    t_start = time.perf_counter()
    RUN_LOG.write_text("")   # clear previous log

    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            ["uv", "run", "denoiser.py"],
            stdout=open(RUN_LOG, "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        proc.wait(timeout=TIMEOUT_SECONDS)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  TIMEOUT after {TIMEOUT_SECONDS}s — treating as crash")
        log_to_tsv({"primary": 0, "raw_baseline": 0, "improvement_pct": 0,
                    "smoothness": 0, "latency_ms": 0},
                   description, "crash", h)
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", "denoiser.py")
        return best_rmse
    except Exception as e:
        print(f"  ERROR launching experiment: {e}")
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", "denoiser.py")
        return best_rmse

    elapsed = time.perf_counter() - t_start
    print(f"  Finished in {elapsed:.0f}s (exit={returncode})")

    # Parse results
    results = parse_results(RUN_LOG)
    if results is None:
        print("  CRASH: could not parse results")
        tail = RUN_LOG.read_text()[-2000:]
        print(f"  Last 2000 chars of log:\n{tail}")
        log_to_tsv({"primary": 0, "raw_baseline": 0, "improvement_pct": 0,
                    "smoothness": 0, "latency_ms": 0},
                   description, "crash", h)
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", "denoiser.py")
        return best_rmse

    primary  = results["primary"]
    all_pass = results["all_pass"]

    print(f"  primary_rmse:    {primary:.6f}  (best so far: {best_rmse:.6f})")
    print(f"  improvement_pct: {results.get('improvement_pct', 0):.2f}%")
    print(f"  all_guardrails:  {all_pass}")

    improved = primary < best_rmse

    if improved and all_pass:
        status   = "keep"
        best_rmse = primary
        print(f"  KEEP ✓  (new best: {best_rmse:.6f})")
    else:
        reason = []
        if not improved:  reason.append("no improvement")
        if not all_pass:  reason.append("guardrail fail")
        status = "discard"
        print(f"  DISCARD ({', '.join(reason)})")
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", "denoiser.py")

    log_to_tsv(results, description, status, h)
    return best_rmse


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Motor Imagery AutoResearch — overnight loop")
    print(f"Branch: {git('branch', '--show-current').stdout.strip()}")
    print(f"Queue:  {len(QUEUE)} experiments (will repeat)")
    print("=" * 60)

    # Establish baseline best_rmse from results.tsv if any runs already done
    best_rmse = float("inf")
    try:
        import csv
        with open("results.tsv") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        kept = [float(r["primary_rmse"]) for r in rows if r.get("status") == "keep"]
        if kept:
            best_rmse = min(kept)
            print(f"Resuming: best RMSE so far = {best_rmse:.6f}\n")
        else:
            print("No previous kept runs — starting fresh\n")
    except Exception:
        print("No results.tsv entries yet\n")

    iteration = 0
    pass_num  = 0

    while True:
        pass_num += 1
        print(f"\n{'*'*60}")
        print(f"Pass {pass_num} through experiment queue")
        print(f"{'*'*60}")

        for src_file, description in QUEUE:
            iteration += 1
            try:
                best_rmse = run_experiment(src_file, description, best_rmse, iteration)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Stopping loop.")
                sys.exit(0)
            except Exception as e:
                print(f"  Unexpected error in experiment: {e}")
                # Try to clean up git state
                try:
                    git("reset", "--hard", "HEAD~1")
                except Exception:
                    pass
                continue

        print(f"\nPass {pass_num} complete. Best RMSE: {best_rmse:.6f}. Starting next pass ...")


if __name__ == "__main__":
    # Handle SIGTERM gracefully
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
