"""Wait for TokenProjector checkpoint, then run local/global compare UI.

Does not touch the training process — only watches for *.pt.

  python wait_and_compare.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

CKA_ROOT = Path(__file__).resolve().parent
CKPT_DIR = CKA_ROOT / "checkpoints" / "token_projector_full1k_silu_3ep"
# Prefer last; fall back to any epoch ckpt so compare can run mid-train.
CANDIDATES = [
    CKPT_DIR / "aligner_last.pt",
    CKPT_DIR / "aligner_epoch001.pt",
]
POLL_SEC = 120
PYTHON = Path(r"C:\Users\FITT\Desktop\MLP\.venv\Scripts\python.exe")
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def find_ckpt() -> Path | None:
    for p in CANDIDATES:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    epochs = sorted(CKPT_DIR.glob("aligner_epoch*.pt"))
    if epochs and epochs[-1].stat().st_size > 1000:
        return epochs[-1]
    return None


def main() -> None:
    print(f"watching {CKPT_DIR}", flush=True)
    while True:
        ckpt = find_ckpt()
        if ckpt is not None:
            break
        print(f"  still waiting… ({time.strftime('%H:%M:%S')})", flush=True)
        time.sleep(POLL_SEC)

    time.sleep(5)
    print(f"found {ckpt} ({ckpt.stat().st_size:,} bytes)", flush=True)
    cmd = [
        str(PYTHON),
        "-u",
        str(CKA_ROOT / "compare_local_global.py"),
        "--config",
        str(CKA_ROOT / "configs" / "eval_token_projector_val.yaml"),
        "--checkpoint",
        str(ckpt),
        "--output",
        str(CKA_ROOT / "results" / "local_global_compare"),
    ]
    print("running:", " ".join(cmd), flush=True)
    log = CKA_ROOT / "results" / "local_global_compare_run.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(CKA_ROOT), stdout=f, stderr=subprocess.STDOUT)
    print(f"compare exit={proc.returncode}  log={log}", flush=True)
    print(f"UI: {CKA_ROOT / 'results' / 'local_global_compare' / 'index.html'}", flush=True)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
