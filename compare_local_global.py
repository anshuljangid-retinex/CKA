"""Decode TokenProjector local vs global vs full on each val image; write compare UI.

  python compare_local_global.py --config configs/eval_token_projector_val.yaml
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path

import torch

CKA_ROOT = Path(__file__).resolve().parent
MLP_ROOT = Path(r"C:\Users\FITT\Desktop\MLP")
for p in (str(CKA_ROOT), str(MLP_ROOT), str(MLP_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluate as mlp_eval  # noqa: E402
from token_projector import TokenProjector  # noqa: E402

from evaluate_token_projector import load_model_token_projector, _resolve_paths  # noqa: E402

BRANCHES = ("local", "global", "full")


def write_compare_html(out_dir: Path, payload: dict) -> Path:
    """Side-by-side local | global | GT viewer (full optional)."""
    data_json = json.dumps(payload, ensure_ascii=False)
    data_json = data_json.replace("<", "\\u003c").replace(">", "\\u003e")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Local vs Global TokenProjector</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  :root {{
    --bg: #eef1f4; --ink: #12151a; --muted: #5a6570; --line: #d0d7de;
    --card: #ffffff; --accent: #0d6e6e; --accent-soft: #d7efef;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh;
    font-family: "DM Sans", system-ui, sans-serif; color: var(--ink);
    background:
      radial-gradient(1200px 500px at 10% -10%, #d9ecec 0%, transparent 55%),
      radial-gradient(900px 400px at 100% 0%, #e8e4f5 0%, transparent 50%),
      var(--bg);
  }}
  .wrap {{ max-width: 1400px; margin: 0 auto; padding: 1.25rem 1.25rem 3rem; }}
  h1 {{ font-size: 1.45rem; margin: 0; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); margin: .35rem 0 1rem; font-size: .9rem; }}
  .metrics {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem;
    margin-bottom: 1rem;
  }}
  @media (max-width: 800px) {{ .metrics {{ grid-template-columns: 1fr; }} }}
  .metric-card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: .85rem 1rem;
  }}
  .metric-card h2 {{ margin: 0; font-size: .78rem; text-transform: uppercase;
    letter-spacing: .06em; color: var(--muted); }}
  .metric-card .nums {{ margin-top: .45rem; font-family: "IBM Plex Mono", monospace;
    font-size: .85rem; line-height: 1.55; }}
  .viewer {{ display: grid; grid-template-columns: 220px 1fr; gap: 1rem; min-height: 70vh; }}
  @media (max-width: 900px) {{ .viewer {{ grid-template-columns: 1fr; }} }}
  .rail, .detail {{
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  }}
  .rail {{ padding: .75rem; overflow: auto; max-height: 80vh; }}
  .rail h2 {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
    color: var(--muted); margin: 0 0 .6rem; }}
  .thumb-btn {{
    display: flex; gap: .55rem; align-items: center; width: 100%;
    border: 1px solid transparent; background: transparent; border-radius: 8px;
    padding: .4rem; cursor: pointer; text-align: left; font: inherit; color: inherit;
    margin-bottom: .35rem;
  }}
  .thumb-btn:hover {{ background: #f6f8fa; }}
  .thumb-btn.active {{ border-color: var(--accent); background: var(--accent-soft); }}
  .thumb-btn img {{
    width: 56px; height: 56px; object-fit: cover; border-radius: 6px;
    border: 1px solid var(--line); background: #eee; flex-shrink: 0;
  }}
  .thumb-btn .name {{
    font-size: .78rem; font-weight: 600; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .thumb-btn .meta {{ font-size: .72rem; color: var(--muted); font-family: "IBM Plex Mono", monospace; }}
  .detail {{ padding: 1rem; overflow: auto; max-height: 80vh; }}
  .detail h3 {{ margin: 0 0 .5rem; font-size: 1.05rem; }}
  .hero {{
    width: 100%; max-height: 320px; object-fit: contain; background: #f6f8fa;
    border: 1px solid var(--line); border-radius: 8px; margin-bottom: .85rem;
  }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .75rem; }}
  @media (max-width: 1000px) {{ .cols {{ grid-template-columns: 1fr; }} }}
  .col {{ border: 1px solid var(--line); border-radius: 10px; padding: .65rem .75rem; background: #fafbfc; }}
  .col h4 {{ margin: 0 0 .4rem; font-size: .82rem; text-transform: uppercase;
    letter-spacing: .04em; color: var(--muted); }}
  .col .score {{ font-family: "IBM Plex Mono", monospace; font-size: .75rem;
    color: var(--muted); margin-bottom: .4rem; }}
  .col pre {{
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-family: "IBM Plex Mono", monospace; font-size: .78rem; line-height: 1.45;
    max-height: 420px; overflow: auto;
  }}
  .col.gt {{ background: #f3faf9; border-color: #b7d9d9; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Local vs Global TokenProjector</h1>
    <p class="sub" id="sub"></p>
    <p class="sub" id="status" style="display:none;color:#9a3412"></p>
  </header>
  <div class="metrics" id="metrics"></div>
  <div class="viewer">
    <aside class="rail">
      <h2>Images</h2>
      <div id="rail"></div>
    </aside>
    <section class="detail" id="detail">
      <p class="sub">Select an image.</p>
    </section>
  </div>
</div>
<script>
const DATA = {data_json};

function fmt(v, d=3) {{
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(d);
}}

function esc(s) {{
  return String(s ?? "").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c]));
}}

function renderMetrics() {{
  const m = DATA.metrics || {{}};
  const branches = ["local", "global", "full"];
  document.getElementById("sub").textContent =
    `${{DATA.n_samples || 0}} images · ${{DATA.checkpoint || "no checkpoint yet"}}`;
  const st = document.getElementById("status");
  if (DATA.status === "waiting_for_checkpoint" || DATA.note) {{
    st.style.display = "block";
    st.textContent = DATA.note || "Waiting for train checkpoint + compare…";
  }}
  document.getElementById("metrics").innerHTML = branches.map(b => {{
    const x = m[b] || {{}};
    return `<div class="metric-card"><h2>${{b}}</h2>
      <div class="nums">CER ${{fmt(x.cer, 4)}} · WER ${{fmt(x.wer, 4)}} · EM ${{fmt(x.exact_match, 4)}}</div>
    </div>`;
  }}).join("");
}}

function renderRail(active) {{
  const rail = document.getElementById("rail");
  rail.innerHTML = (DATA.samples || []).map((s, i) => {{
    const cls = i === active ? "thumb-btn active" : "thumb-btn";
    const img = s.image_rel ? `<img src="${{esc(s.image_rel)}}" alt=""/>` : "";
    return `<button type="button" class="${{cls}}" data-i="${{i}}">
      ${{img}}
      <div><div class="name">${{esc(s.name)}}</div>
      <div class="meta">L ${{fmt(s.local?.cer)}} · G ${{fmt(s.global?.cer)}}</div></div>
    </button>`;
  }}).join("");
  rail.querySelectorAll("button").forEach(btn => {{
    btn.addEventListener("click", () => show(Number(btn.dataset.i)));
  }});
}}

function col(title, body, score, gt) {{
  return `<div class="col${{gt ? " gt" : ""}}">
    <h4>${{esc(title)}}</h4>
    ${{score ? `<div class="score">${{esc(score)}}</div>` : ""}}
    <pre>${{esc(body)}}</pre>
  </div>`;
}}

function show(i) {{
  const s = DATA.samples[i];
  if (!s) return;
  renderRail(i);
  const L = s.local || {{}}, G = s.global || {{}}, F = s.full || {{}};
  document.getElementById("detail").innerHTML = `
    <h3>${{esc(s.name)}}</h3>
    ${{s.image_rel ? `<img class="hero" src="${{esc(s.image_rel)}}" alt=""/>` : ""}}
    <div class="cols">
      ${{col("Local prediction", L.hypothesis,
        `CER=${{fmt(L.cer)}} WER=${{fmt(L.wer)}} EM=${{fmt(L.exact_match, 0)}}`)}}
      ${{col("Global prediction", G.hypothesis,
        `CER=${{fmt(G.cer)}} WER=${{fmt(G.wer)}} EM=${{fmt(G.exact_match, 0)}}`)}}
      ${{col("Ground truth", s.reference || "", null, true)}}
    </div>
    <div class="cols" style="margin-top:.75rem; grid-template-columns:1fr">
      ${{col("Full (local+global) prediction", F.hypothesis,
        `CER=${{fmt(F.cer)}} WER=${{fmt(F.wer)}} EM=${{fmt(F.exact_match, 0)}}`)}}
    </div>`;
}}

renderMetrics();
show(0);
</script>
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


@torch.no_grad()
def compare(cfg: dict, *, out_dir: Path, branches: tuple[str, ...] = BRANCHES) -> dict:
    device = mlp_eval.resolve_device(str(cfg.get("device", "auto")))
    dtype = mlp_eval.resolve_dtype(str(cfg.get("dtype", "bfloat16")), device)
    ckpt_arg = Path(cfg["checkpoint"])
    ckpt_path = mlp_eval.find_checkpoint(ckpt_arg)
    print(f"checkpoint: {ckpt_path}")
    print(f"device: {device}")

    model = load_model_token_projector(cfg, device, dtype, ckpt_path)
    if not isinstance(model.aligner, TokenProjector):
        raise TypeError(f"expected TokenProjector, got {type(model.aligner)}")

    ds = mlp_eval.build_eval_dataset(cfg, model)
    if len(ds) == 0:
        raise RuntimeError("eval set empty")
    print(f"samples: {len(ds)}  branches: {branches}")

    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    max_vision_tokens = cfg.get("max_vision_tokens")
    prompt = mlp_eval.resolve_generate_prompt(cfg.get("prompt", "ocr_layout"))

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    samples: list[dict] = []
    for i in range(len(ds)):
        pixel_values, _labels, grid = ds[i]
        if grid.ndim == 1:
            grid = grid.unsqueeze(0)
        ref = ds.text_at(i) or ""
        name = mlp_eval.sample_name(ds, i)
        safe = Path(name).stem or f"sample_{i}"

        pil = mlp_eval.sample_pil(ds, i)
        image_rel = None
        if pil is not None:
            img_path = img_dir / f"{safe}.jpg"
            pil.convert("RGB").save(img_path, quality=90)
            image_rel = f"images/{safe}.jpg"

        row: dict = {"name": name, "reference": ref, "image_rel": image_rel}
        for branch in branches:
            model.aligner.branch = branch
            hyp = model.generate(
                pixel_values=pixel_values,
                image_grid_thw=grid,
                max_new_tokens=max_new_tokens,
                max_vision_tokens=max_vision_tokens,
                prompt=prompt,
            )[0]
            c = mlp_eval.cer(ref, hyp)
            w = mlp_eval.wer(ref, hyp)
            e = mlp_eval.exact_match(ref, hyp)
            row[branch] = {
                "hypothesis": hyp,
                "cer": c,
                "wer": w,
                "exact_match": e,
            }
            print(
                f"[{i + 1}/{len(ds)}] {branch:6s} CER={c:.3f} WER={w:.3f} EM={e:.0f}  {name}"
            )
        model.aligner.branch = "full"
        samples.append(row)

        # per-image text dump
        sample_dir = out_dir / "per_image" / safe
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "gt.txt").write_text(ref, encoding="utf-8")
        for branch in branches:
            (sample_dir / f"{branch}.txt").write_text(
                row[branch]["hypothesis"] or "", encoding="utf-8"
            )

    metrics = {}
    for b in branches:
        metrics[b] = {
            "cer": sum(s[b]["cer"] for s in samples) / len(samples),
            "wer": sum(s[b]["wer"] for s in samples) / len(samples),
            "exact_match": sum(s[b]["exact_match"] for s in samples) / len(samples),
        }

    payload = {
        "n_samples": len(samples),
        "checkpoint": str(ckpt_path),
        "branches": list(branches),
        "metrics": metrics,
        "samples": samples,
        "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
    }
    json_path = out_dir / "predictions.json"
    # keep full text in JSON; UI embeds this — truncate nothing for offline viewing
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path = write_compare_html(out_dir, payload)
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    for b, m in metrics.items():
        print(f"  {b:6s} CER={m['cer']:.4f} WER={m['wer']:.4f} EM={m['exact_match']:.4f}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TokenProjector local vs global OCR")
    parser.add_argument(
        "--config",
        type=Path,
        default=CKA_ROOT / "configs" / "eval_token_projector_val.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=CKA_ROOT / "results" / "local_global_compare",
    )
    parser.add_argument(
        "--branches",
        default="local,global,full",
        help="comma list: local,global,full",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else (CKA_ROOT / args.config).resolve()
    cfg = mlp_eval.load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    cfg = _resolve_paths(cfg)
    if args.checkpoint is not None:
        ck = args.checkpoint
        cfg["checkpoint"] = str(ck if ck.is_absolute() else (CKA_ROOT / ck).resolve())

    out_dir = args.output if args.output.is_absolute() else (CKA_ROOT / args.output).resolve()
    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())
    for b in branches:
        if b not in BRANCHES:
            raise SystemExit(f"unknown branch {b!r}; choose from {BRANCHES}")
    compare(cfg, out_dir=out_dir, branches=branches)


if __name__ == "__main__":
    main()
