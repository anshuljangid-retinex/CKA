"""Linear-kernel CKA between frozen TeleOCR vision embeds and Chandra text embeds.

Diagnostic only — no training. Reuses MLP loaders/model.

  python compute_cka.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CKA_ROOT = Path(__file__).resolve().parent
MLP_ROOT = Path(r"C:\Users\FITT\Desktop\MLP")
if str(MLP_ROOT) not in sys.path:
    sys.path.insert(0, str(MLP_ROOT))

from data.synthdocs_dataset import SynthDocsDataset  # noqa: E402
from models.combined_pipeline import CombinedOCRModel  # noqa: E402

DEFAULT_DATA = MLP_ROOT / "Retinex-AIGCsynthdocs-warp-slice-1k"
DEFAULT_OUT = CKA_ROOT / "results" / "cka_report.json"


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> float:
    """Linear CKA via centered features (avoids n×n Gram when n is large).

    X, Y: [n, d] float tensors (same n). Returns scalar in [0, 1] (ideally).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"expected 2D tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row mismatch: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        raise ValueError("need at least 2 rows for CKA")

    x = x.double() - x.double().mean(dim=0, keepdim=True)
    y = y.double() - y.double().mean(dim=0, keepdim=True)
    xtx = x.T @ x
    yty = y.T @ y
    xty = x.T @ y
    hsic_xy = (xty * xty).sum()
    hsic_xx = (xtx * xtx).sum()
    hsic_yy = (yty * yty).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy).clamp_min(eps)
    return float((hsic_xy / denom).item())


def mean_pool(tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """tokens [T, D] or [B, T, D]; mask optional [T] or [B, T] with 1=valid."""
    if tokens.ndim == 2:
        if mask is None:
            return tokens.mean(dim=0)
        m = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * m).sum(dim=0) / m.sum().clamp_min(1.0)
    if mask is None:
        return tokens.mean(dim=1)
    m = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * m).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1.0)


@torch.no_grad()
def collect_embeddings(
    model: CombinedOCRModel,
    dataset: SynthDocsDataset,
    *,
    device: torch.device,
    max_vision_tokens: int,
    max_target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Return global_v [N,Xv], global_t [N,Yt], list of local patch tensors [Ti,Xv]."""
    embed = model.language_backbone.get_input_embeddings()
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0

    globals_v: list[torch.Tensor] = []
    globals_t: list[torch.Tensor] = []
    locals_v: list[torch.Tensor] = []

    model.eval()
    n = len(dataset)
    for i in range(n):
        pixel_values, label_ids, image_grid_thw = dataset[i]
        # Dataset returns squeezed single-sample tensors; encode expects batched.
        pv = pixel_values.unsqueeze(0).to(device)
        grid = image_grid_thw.unsqueeze(0).to(device)
        vision_embeds, vision_mask = model._encode(pv, grid)
        if max_vision_tokens and vision_embeds.shape[1] > max_vision_tokens:
            vision_embeds = vision_embeds[:, :max_vision_tokens]
            vision_mask = vision_mask[:, :max_vision_tokens]

        patches = vision_embeds[0].float().cpu()  # [T, X]
        mask = vision_mask[0].cpu()
        valid = patches[mask.bool()]
        if valid.shape[0] == 0:
            continue

        ids = label_ids[:max_target_length].unsqueeze(0).to(device)
        text_emb = embed(ids).float()[0].cpu()  # [L, Y]
        if pad_id is not None:
            tmask = ids[0].cpu() != pad_id
            text_emb = text_emb[tmask]
        if text_emb.shape[0] == 0:
            continue

        locals_v.append(valid)
        globals_v.append(valid.mean(dim=0))
        globals_t.append(text_emb.mean(dim=0))

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  embeds {i + 1}/{n}", flush=True)

    if len(globals_v) != len(globals_t):
        # Drop unpaired if a sample had empty text after pad filter (should not happen).
        m = min(len(globals_v), len(globals_t))
        globals_v, globals_t, locals_v = globals_v[:m], globals_t[:m], locals_v[:m]

    return torch.stack(globals_v, dim=0), torch.stack(globals_t, dim=0), locals_v


def main() -> None:
    parser = argparse.ArgumentParser(description="CKA diagnostic: TeleOCR vs Chandra embeds")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_vision_tokens", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teleocr_id", default="StarDoc-AI/TeleOCR")
    parser.add_argument("--chandra_id", default="datalab-to/chandra-ocr-2")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[args.dtype.lower()]
    if device.type == "cpu":
        dtype = torch.float32

    print(f"loading CombinedOCRModel on {device} dtype={dtype} …", flush=True)
    model = CombinedOCRModel.from_pretrained(
        teleocr_id=args.teleocr_id,
        chandra_id=args.chandra_id,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"loading dataset max_samples={args.max_samples} from {args.data_root} …", flush=True)
    dataset = SynthDocsDataset(
        root=args.data_root,
        processor=model.processor,
        tokenizer=model.tokenizer,
        max_samples=args.max_samples,
        max_target_length=args.max_target_length,
        split=args.split,
        text_field="teacher_raw_html",
    )
    print(f"n={len(dataset)}", flush=True)

    print("collecting embeddings …", flush=True)
    g_v, g_t, locals_v = collect_embeddings(
        model,
        dataset,
        device=device,
        max_vision_tokens=args.max_vision_tokens,
        max_target_length=args.max_target_length,
    )

    # Local: all valid patches; each paired with that sample's pooled text embed.
    local_rows_v: list[torch.Tensor] = []
    local_rows_t: list[torch.Tensor] = []
    for patches, t_vec in zip(locals_v, g_t):
        local_rows_v.append(patches)
        local_rows_t.append(t_vec.unsqueeze(0).expand(patches.shape[0], -1))
    x_local = torch.cat(local_rows_v, dim=0)
    y_local = torch.cat(local_rows_t, dim=0)

    print(f"computing CKA  global n={g_v.shape[0]}  local n={x_local.shape[0]} …", flush=True)
    cka_global = linear_cka(g_v, g_t)
    cka_local = linear_cka(x_local, y_local)

    report = {
        "cka_global": cka_global,
        "cka_local": cka_local,
        "n_samples": int(g_v.shape[0]),
        "n_local_tokens": int(x_local.shape[0]),
        "vision_dim": int(g_v.shape[1]),
        "text_dim": int(g_t.shape[1]),
        "max_samples_requested": args.max_samples,
        "max_vision_tokens": args.max_vision_tokens,
        "max_target_length": args.max_target_length,
        "split": args.split,
        "seed": args.seed,
        "data_root": str(args.data_root),
        "teleocr_id": args.teleocr_id,
        "chandra_id": args.chandra_id,
        "text_field": "teacher_raw_html",
        "kernel": "linear",
        "note": "cka_local pairs each patch token with its sample's mean-pooled text embedding",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nCKA global={cka_global:.4f}  local={cka_local:.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
