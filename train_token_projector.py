"""Train TokenProjector only (TeleOCR + Chandra frozen). Reuses MLP Stage-1 loop.

  python train_token_projector.py --config configs/token_projector_full1k_silu_3ep.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

CKA_ROOT = Path(__file__).resolve().parent
MLP_ROOT = Path(r"C:\Users\FITT\Desktop\MLP")
for p in (str(CKA_ROOT), str(MLP_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import collate_stage2  # noqa: E402
from data.synthdocs_dataset import SynthDocsDataset  # noqa: E402
from models.combined_pipeline import CombinedOCRModel, count_trainable_params, resolve_ocr_prompt  # noqa: E402
from token_projector import TokenProjector  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device):
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    dtype = table[str(name).lower()]
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
        return torch.float32
    return dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_projector_checkpoint(
    path: Path,
    projector: torch.nn.Module,
    *,
    epoch: int,
    step: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "aligner": projector.state_dict(),  # same key as MLP eval loaders
            "token_projector": projector.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
            "aligner_kind": "TokenProjector",
        },
        path,
    )
    print(f"saved TokenProjector checkpoint -> {path}", flush=True)


def build_model(cfg: dict, dataset: SynthDocsDataset, device: torch.device, dtype) -> CombinedOCRModel:
    acfg = cfg.get("aligner") or {}
    model = CombinedOCRModel.from_pretrained(
        teleocr_id=cfg.get("teleocr_id", "StarDoc-AI/TeleOCR"),
        chandra_id=cfg.get("chandra_id", "datalab-to/chandra-ocr-2"),
        torch_dtype=dtype,
        device_map=None,
    )
    model.aligner = TokenProjector(
        input_dim=model.aligner.input_dim,
        output_dim=model.aligner.output_dim,
        hidden_dim=int(acfg.get("hidden_dim", 2048)),
        dropout=float(acfg.get("dropout", 0.0)),
        activation=str(acfg.get("activation", "silu")),
    )
    dataset.processor = model.processor
    dataset.image_processor = getattr(model.processor, "image_processor", model.processor)
    dataset.tokenizer = model.tokenizer

    model.to(device)
    model.aligner.to(device=device, dtype=torch.float32)
    model.train()
    return model


def train(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    dtype = resolve_dtype(str(cfg.get("dtype", "bfloat16")), device)

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints/token_projector"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = CKA_ROOT / ckpt_dir

    data_root = cfg.get("data_root") or (MLP_ROOT / "Retinex-AIGCsynthdocs-warp-slice-1k")
    max_samples = cfg.get("max_samples")
    if max_samples is not None and int(max_samples) <= 0:
        max_samples = None

    dataset = SynthDocsDataset(
        data_root,
        max_samples=max_samples,
        max_target_length=int(cfg.get("max_target_length", 512)),
        teleocr_id=cfg.get("teleocr_id", "StarDoc-AI/TeleOCR"),
        chandra_id=cfg.get("chandra_id", "datalab-to/chandra-ocr-2"),
        split=cfg.get("split", "train"),
        text_field="teacher_raw_html",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate_stage2,
    )
    print(f"train samples={len(dataset)} batches/epoch={len(loader)} device={device}", flush=True)

    model = build_model(cfg, dataset, device, dtype)
    trainable, _ = count_trainable_params(model)
    proj_n = sum(p.numel() for p in model.aligner.parameters())
    if trainable != proj_n:
        raise RuntimeError(f"expected only TokenProjector trainable ({proj_n}), got {trainable}")

    lr = float(cfg.get("lr", 1e-4))
    optimizer = torch.optim.AdamW(
        model.aligner.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    grad_clip = float(cfg.get("grad_clip", 1.0))
    log_every = max(1, int(cfg.get("log_every", 20)))
    epochs = int(cfg.get("epochs", 3))
    max_vision_tokens = cfg.get("max_vision_tokens")
    train_prompt = resolve_ocr_prompt(cfg.get("prompt", "ocr_layout"))
    print(
        f"train prompt: "
        f"{'(none)' if train_prompt is None else train_prompt[:72] + ('…' if len(train_prompt) > 72 else '')}",
        flush=True,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "train_config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_sum = 0.0
        running_count = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for pixel_values, grid, labels, label_mask in zip(
                batch["pixel_values"],
                batch["image_grid_thw"],
                batch["labels"],
                batch["label_mask"],
            ):
                if grid.ndim == 1:
                    grid = grid.unsqueeze(0)
                losses.append(
                    model.compute_lm_loss(
                        pixel_values=pixel_values,
                        image_grid_thw=grid,
                        target_ids=labels,
                        target_mask=label_mask,
                        max_vision_tokens=max_vision_tokens,
                        prompt=train_prompt,
                    )
                )
            loss = torch.stack(losses).mean()
            global_step += 1

            if not torch.isfinite(loss):
                print(
                    f"WARNING: non-finite loss at epoch={epoch} step={global_step} — skip step",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.aligner.parameters(), grad_clip)
            optimizer.step()

            loss_val = float(loss.item())
            running_sum += loss_val
            running_count += 1
            if global_step % log_every == 0:
                print(f"epoch={epoch} step={global_step} loss={loss_val:.4f} lr={lr}", flush=True)

        avg = running_sum / max(1, running_count)
        print(f"epoch={epoch} done avg_loss={avg:.4f} (finite_steps={running_count})", flush=True)
        save_projector_checkpoint(
            ckpt_dir / f"aligner_epoch{epoch:03d}.pt",
            model.aligner,
            epoch=epoch,
            step=global_step,
            cfg=cfg,
        )

    save_projector_checkpoint(
        ckpt_dir / "aligner_last.pt",
        model.aligner,
        epoch=epochs,
        step=global_step,
        cfg=cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TokenProjector (aligner-only)")
    parser.add_argument(
        "--config",
        type=Path,
        default=CKA_ROOT / "configs" / "token_projector_full1k_silu_3ep.yaml",
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not cfg_path.is_absolute():
        cfg_path = (CKA_ROOT / cfg_path).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {args.config}")
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    train(cfg)


if __name__ == "__main__":
    main()
