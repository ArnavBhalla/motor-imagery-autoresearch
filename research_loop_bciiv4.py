"""
Autonomous research loop for BCI Competition IV Dataset 4 (real ECoG).

Mirrors research_loop.py but targets experiments_real/ and results_bciiv4.tsv.
Each experiment file imports from prepare_bciiv4 instead of prepare.

Usage:
    nohup uv run research_loop_bciiv4.py > research_bciiv4.log 2>&1 &
    tail -f research_bciiv4.log
    kill %2
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

EXPERIMENTS_DIR = Path("experiments_real")

QUEUE = [
    ("r28_supervised_small.py", "real CNN small ch=64 d=4 ECoG-HG"),
    ("r28_supervised_small.py",   "real CNN small ch=64 d=4 ECoG-HG"),
    ("r27_supervised_big.py",     "real CNN big ch=128 d=6 ECoG-HG"),
    ("r31_mamba.py",              "real Mamba SSM d=64 d_state=16 n_layers=4 ECoG-HG"),
    ("r29_transformer_smooth.py", "real transformer smooth lambda_s=0.5 d=64 h=4 l=4 ECoG-HG"),
]

TIMEOUT_SECONDS = 1200   # 20 min — decoder + denoiser training both happen
RUN_LOG  = Path("run_bciiv4.log")
RESULTS  = Path("results_bciiv4.tsv")
DENOISER = Path("denoiser_real.py")


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True)


def short_hash():
    r = git("rev-parse", "--short", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else "xxxxxxx"


def parse_results(log_path: Path):
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
        f"{results.get('raw_baseline', 0) or 0:.6f}",
        f"{results.get('improvement_pct', 0) or 0:.2f}",
        f"{results.get('smoothness', 0) or 0:.4f}",
        f"{results.get('latency_ms', 0) or 0:.1f}",
        status,
        description,
    ])
    with open(RESULTS, "a") as f:
        f.write(row + "\n")
    print(f"  → logged ({status}): {description}")


def run_experiment(src_file: str, description: str, best_rmse: float, iteration: int):
    print(f"\n{'='*60}")
    print(f"[real iter {iteration:03d}] {description}")
    print(f"{'='*60}")

    src_path = EXPERIMENTS_DIR / src_file
    if not src_path.exists():
        print(f"  SKIP: {src_path} not found")
        return best_rmse

    shutil.copy(src_path, DENOISER)
    print(f"  Applied: {src_path} → {DENOISER}")

    git("add", str(DENOISER))
    commit_msg = f"[real iter {iteration:03d}] {description}"
    r = git("commit", "-m", commit_msg)
    if r.returncode != 0 and "nothing to commit" not in r.stdout + r.stderr:
        git("commit", "--allow-empty", "-m", commit_msg)
    h = short_hash()
    print(f"  Commit: {h}")

    print(f"  Running (timeout={TIMEOUT_SECONDS}s) ...")
    t_start = time.perf_counter()
    RUN_LOG.write_text("")

    try:
        env  = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            ["uv", "run", str(DENOISER)],
            stdout=open(RUN_LOG, "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        proc.wait(timeout=TIMEOUT_SECONDS)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  TIMEOUT after {TIMEOUT_SECONDS}s")
        log_to_tsv({"primary": 0, "raw_baseline": 0, "improvement_pct": 0,
                    "smoothness": 0, "latency_ms": 0},
                   description, "crash", h)
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", str(DENOISER))
        return best_rmse
    except Exception as e:
        print(f"  ERROR: {e}")
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", str(DENOISER))
        return best_rmse

    elapsed = time.perf_counter() - t_start
    print(f"  Finished in {elapsed:.0f}s (exit={returncode})")

    results = parse_results(RUN_LOG)
    if results is None:
        print("  CRASH: could not parse results")
        tail = RUN_LOG.read_text()[-2000:]
        print(f"  Last 2000 chars:\n{tail}")
        log_to_tsv({"primary": 0, "raw_baseline": 0, "improvement_pct": 0,
                    "smoothness": 0, "latency_ms": 0},
                   description, "crash", h)
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", str(DENOISER))
        return best_rmse

    primary  = results["primary"]
    all_pass = results["all_pass"]

    print(f"  primary_rmse:    {primary:.6f}  (best so far: {best_rmse:.6f})")
    print(f"  improvement_pct: {results.get('improvement_pct', 0):.2f}%")
    print(f"  all_guardrails:  {all_pass}")

    improved = primary < best_rmse

    if improved and all_pass:
        status    = "keep"
        best_rmse = primary
        print(f"  KEEP ✓  (new best: {best_rmse:.6f})")
    else:
        reason = []
        if not improved: reason.append("no improvement")
        if not all_pass: reason.append("guardrail fail")
        status = "discard"
        print(f"  DISCARD ({', '.join(reason)})")
        git("reset", "--mixed", "HEAD~1")
        git("checkout", "--", str(DENOISER))

    log_to_tsv(results, description, status, h)
    return best_rmse


def main():
    print("=" * 60)
    print("BCI IV Dataset 4 — real ECoG autoresearch loop")
    print(f"Branch: {git('branch', '--show-current').stdout.strip()}")
    print(f"Queue:  {len(QUEUE)} experiments (will repeat)")
    print("=" * 60)

    # Write TSV header if new file
    if not RESULTS.exists():
        with open(RESULTS, "w") as f:
            f.write("commit\tprimary_rmse\traw_baseline\timprovement_pct\t"
                    "smoothness\tlatency_ms\tstatus\tdescription\n")

    best_rmse = float("inf")
    try:
        import csv
        with open(RESULTS) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        kept = [float(r["primary_rmse"]) for r in rows if r.get("status") == "keep"]
        if kept:
            best_rmse = min(kept)
            print(f"Resuming: best RMSE so far = {best_rmse:.6f}\n")
        else:
            print("No previous kept runs — starting fresh\n")
    except Exception:
        print("No results yet\n")

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
                print("\nInterrupted. Stopping.")
                sys.exit(0)
            except Exception as e:
                print(f"  Unexpected error: {e}")
                try:
                    git("reset", "--hard", "HEAD~1")
                except Exception:
                    pass
                continue

        print(f"\nPass {pass_num} complete. Best RMSE: {best_rmse:.6f}. "
              f"Starting next pass ...")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
