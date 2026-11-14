# CKA + TokenProjector summary

## CKA diagnostic (frozen TeleOCR vs Chandra embeds)

Dataset: `Retinex-AIGCsynthdocs-warp-slice-1k`, 500 train samples, linear kernel.  
Report: [`results/cka_report.json`](results/cka_report.json)

| Metric | Score |
|--------|------:|
| **CKA global** (mean-pooled vision vs mean-pooled text) | **0.298** |
| **CKA local** (patch tokens vs pooled text) | **0.121** |

Interpretation: spaces are only weakly alignable; global structure is stronger than patch-level. A better projector can help a bit, but large gains are unlikely from the aligner alone.

## TokenProjector train (fast rerun)

Hung full-1k×3ep run was killed (~18h, no `.pt`). Restarted with a workable config matching the MLP silu 128×1 prompted scale:

- config: `configs/token_projector_128x1_fast.yaml`
- train: 128 samples × 1 epoch, `max_vision_tokens=256`, `max_target_length=256`, silu, `ocr_layout` prompt
- avg_loss: **1.7694**
- checkpoint: `checkpoints/token_projector_128x1_fast/aligner_last.pt`

## Eval on val (n=21) + BLEURT

| Run | CER ↓ | WER ↓ | EM | BLEURT ↑ |
|-----|------:|------:|---:|---------:|
| **MLP** silu 128×1 prompted (baseline) | 0.901 | 1.546 | 0 | −1.353 |
| **TokenProjector** 128×1 fast | **0.892** | **0.976** | 0 | −1.436 |

Sources:
- MLP: `MLP/results/stage1_synthdocs_silu_128x1_prompted_eval.json`
- TokenProjector: [`results/token_projector_128x1_fast_eval.json`](results/token_projector_128x1_fast_eval.json)

## Takeaways

1. CKA (~0.30 global / ~0.12 local) says vision↔text spaces are only partially compatible before any train.
2. TokenProjector vs MLP on the same 128×1 scale: **slightly better CER**, **clearly better WER**, BLEURT roughly similar (slightly worse).
3. Absolute OCR quality is still poor (CER ~0.89) — projector upgrade alone is not enough for usable HTML OCR.
4. Full 979×3 train is too slow on this 20GB Ada box without shorter sequences / fast attn kernels; use the fast config for iteration.
