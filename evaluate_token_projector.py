"""Evaluate TokenProjector checkpoint with MLP evaluate.py + BLEURT.

  python evaluate_token_projector.py --config configs/eval_token_projector_val.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

CKA_ROOT = Path(__file__).resolve().parent
MLP_ROOT = Path(r"C:\Users\FITT\Desktop\MLP")
for p in (str(CKA_ROOT), str(MLP_ROOT), str(MLP_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluate as mlp_eval  # noqa: E402
from models.combined_pipeline import CombinedOCRModel  # noqa: E402
from token_projector import TokenProjector  # noqa: E402


def load_model_token_projector(cfg: dict, device: torch.device, dtype, ckpt_path: Path):
    acfg = cfg.get("aligner") or {}
    model = CombinedOCRModel.from_pretrained(
        teleocr_id=cfg.get("teleocr_id", "StarDoc-AI/TeleOCR"),
        chandra_id=cfg.get("chandra_id", "datalab-to/chandra-ocr-2"),
        torch_dtype=dtype,
    )
    model.aligner = TokenProjector(
        input_dim=model.aligner.input_dim,
        output_dim=model.aligner.output_dim,
        hidden_dim=int(acfg.get("hidden_dim", 2048)),
        dropout=float(acfg.get("dropout", 0.0)),
        activation=str(acfg.get("activation", "silu")),
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = blob["aligner"] if isinstance(blob, dict) and "aligner" in blob else blob
    model.aligner.load_state_dict(state)
    model.to(device)
    model.aligner.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _resolve_paths(cfg: dict) -> dict:
    """Make relative checkpoint/output paths resolve under CKA_ROOT."""
    for key in ("checkpoint", "output", "report_dir"):
        val = cfg.get(key)
        if not val:
            continue
        p = Path(val)
        if not p.is_absolute():
            cfg[key] = str((CKA_ROOT / p).resolve())
    bleurt = cfg.get("bleurt_checkpoint")
    if bleurt:
        bp = Path(bleurt)
        if not bp.is_absolute():
            cfg["bleurt_checkpoint"] = str((MLP_ROOT / bp).resolve())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval TokenProjector (CER/WER/EM/BLEURT)")
    parser.add_argument(
        "--config",
        type=Path,
        default=CKA_ROOT / "configs" / "eval_token_projector_val.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--no_bleurt", action="store_true", default=False)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (CKA_ROOT / args.config).resolve()
    cfg = mlp_eval.load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    cfg = _resolve_paths(cfg)
    if args.checkpoint is not None:
        ck = args.checkpoint
        cfg["checkpoint"] = str(ck if ck.is_absolute() else (CKA_ROOT / ck).resolve())
    if args.no_bleurt:
        cfg["bleurt"] = False
    else:
        cfg["bleurt"] = True

    # Point MLP evaluate helpers at CKA for relative outputs when cwd differs.
    mlp_eval.load_model = load_model_token_projector

    # Windows cp1252 consoles choke on ↓/↑ in print_table — keep ASCII.
    _orig_print_table = mlp_eval.print_table

    def _safe_print_table(metrics: dict) -> None:
        import sys

        rows = [
            ("n", metrics["n_samples"]),
            ("CER", f"{metrics['cer']:.4f}"),
            ("WER", f"{metrics['wer']:.4f}"),
            ("EM", f"{metrics['exact_match']:.4f}"),
        ]
        if metrics.get("bleurt") is not None:
            rows.append(("BLEURT", f"{metrics['bleurt']:.4f}"))
        rows.append(("checkpoint", metrics.get("checkpoint", "")))
        width = max(len(str(k)) for k, _ in rows)
        for k, v in rows:
            line = f"{str(k):<{width}}  {v}"
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                sys.stdout.buffer.write((line + "\n").encode("utf-8", errors="replace"))

    mlp_eval.print_table = _safe_print_table
    mlp_eval.evaluate(cfg, save_images=args.save_images)


if __name__ == "__main__":
    main()
