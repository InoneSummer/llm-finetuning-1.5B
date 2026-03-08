# UI Screenshot → React Component: A 2-Stage Fine-tuning Pipeline

> Fine-tuning open-source vision and code models to convert UI designs into working code

---

## Motivation

No-code and low-code tools are reshaping how software is built. The core promise of tools like Figma to Code is simple: **take a visual design and produce working code automatically.**

This project implements that pipeline end-to-end using open-source model fine-tuning:

1. A vision-language model learns to extract HTML structure from UI screenshots
2. A code-specialized model learns to generate clean React + Tailwind components from that structure

The goal is not to build a production tool, but to demonstrate that this pipeline is learnable with relatively small models, public datasets, and accessible hardware.

---

## Pipeline Structure

```
[Stage 1] UI Screenshot → HTML Structure
          Model: Qwen2-VL-7B (multimodal)
          Hardware: RunPod A100 80GB + Unsloth
          Dataset: ronantakizawa/webui

          ↓

[Stage 2] Natural Language / HTML Structure → React/TSX Code
          Model: Qwen2.5-Coder-1.5B (code-specialized)
          Hardware: Mac Mini M4 Pro, 64GB (MLX)
          Dataset: cfahlgren1/react-code-instructions

          ↓

[Validation] LangGraph + AST Pipeline
          Syntax validation → Auto-correction → Quality scoring
```

---

## Project A — UI Screenshot → HTML

Fine-tunes Qwen2-VL-7B to extract HTML structure from UI screenshots.

| Item | Detail |
|------|--------|
| Model | `Qwen/Qwen2-VL-7B-Instruct` (4-bit) |
| Dataset | `ronantakizawa/webui` |
| Training samples | 2,000 (HTML ≤ 4,000 chars) |
| Method | LoRA (rank=16, alpha=32) |
| Framework | Unsloth + HuggingFace TRL |
| Hardware | RunPod A100 80GB PCIe |
| Training time | ~32 minutes |
| Final loss | 0.7341 |

### Before / After

**Before (base model)** — generic Tailwind boilerplate, unrelated to the screenshot:
```html
<html>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<body class="bg-gray-100 font-sans">
    <div class="flex flex-col min-h-screen">
        <header class="bg-gray-800 text-white p-4">
```

**After (fine-tuned)** — structured HTML matching the actual UI:
```html
<!DOCTYPE html>
<html lang="en">
 <head>
  <meta charset="utf-8"/>
  <title>Markdown Editor</title>
  <link href="https://fonts.googleapis.com" rel="preconnect"/>
```

See [`project-a-vision/README.md`](./project-a-vision/README.md) for full details.

---

## Project B — Natural Language → React/TSX

Fine-tunes Qwen2.5-Coder-1.5B to generate clean React + Tailwind components.

| Item | Detail |
|------|--------|
| Model | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` |
| Dataset | `cfahlgren1/react-code-instructions` |
| Training samples | 500 train / 300 val |
| Method | LoRA |
| Framework | mlx-lm (Apple MLX) |
| Final loss | ~0.28 |

### Before / After

| | Base Model | Fine-tuned |
|---|---|---|
| Library usage | Mixes Tailwind with `@chakra-ui/react` | Pure Tailwind CSS only |
| TypeScript | No type definitions | Proper interfaces and typed props |
| Code quality | Inconsistent, off-topic features | Focused, production-ready |

See [`project-b-react/README.md`](./project-b-react/README.md) for full details.

---

## Code Quality Validation Pipeline

A LangGraph pipeline that automatically validates and corrects generated React/TSX code.

```
Input: Generated React/TSX code
      ↓
[parse_code]   JSX syntax validation via Babel AST
      ↓ on error, repeat up to 3 times
[fix_code]     Auto-correction via local LLM (Ollama)
      ↓ on pass
[score_code]   Quality scoring based on AST metrics
      ↓
Output: { valid, score, components, jsx_elements, hooks, attempts }
```

**AST-based quality score** (objective, reproducible — no LLM-as-judge):
```python
score = components * 10 + jsx_elements * 2 + hooks * 5 + (20 if valid else 0)
```

---

## Project Structure

```
/
├── ast_pipeline/               # AST validation pipeline
│   ├── parse.js                # Babel-based JSX syntax validator
│   ├── pipeline.py             # LangGraph validation pipeline
│   └── package.json
├── mlx_ready_dataset/          # Full Parquet dataset (Stage 1)
├── mlx_ready_dataset_truncated/ # Filtered Parquet dataset (used for training)
├── project-a-vision/           # Stage 1: Screenshot → HTML
│   ├── get_data.py
│   ├── prepare_data.py
│   ├── convert_data.py
│   ├── truncate_data.py
│   ├── train.py
│   ├── inference.py
│   ├── adapters_qwen/          # Trained LoRA adapters
│   └── README.md
├── project-b-react/            # Stage 2: NL → React/TSX
│   ├── preprocess_b.py
│   ├── checkpoints_b2/         # Trained LoRA adapters
│   ├── data/
│   ├── examples/
│   └── README.md
├── TROUBLESHOOTING.md
└── README.md
```

---

## Future Work

### DPO (Direct Preference Optimization)

Currently only SFT has been applied. The next step is to build a DPO dataset using AST-based automatic evaluation — no LLM-as-judge required.

```
Generate multiple HTML outputs from the SFT model for the same image
        ↓
Auto-rank by AST score → construct chosen / rejected pairs
        ↓
DPO fine-tuning to learn preference for higher-quality code
```

### Animation and Interaction Support

This project focuses on static HTML structure. However, the `ronantakizawa/webui` dataset includes a `has_animations` column and a separate `css` column — meaning **training data for animation-aware code generation already exists.**

The next step is to extend Stage 1 to generate HTML + CSS animation code by:
- Filtering `has_animations=True` samples
- Including the `css` column as part of the training target
- Teaching the model to generate `@keyframes`, `transition`, and `animation` properties alongside structure

This would bring the pipeline significantly closer to the full design-to-interactive-code vision.

### End-to-End Stage Connection

The two stages are currently fine-tuned independently. Connecting Stage 1 output (HTML) as Stage 2 input would complete the fully integrated pipeline.

### Domain Expansion

Current: Screenshot → static HTML  
Goal: Design tool JSON (layer tree + event graph) → React/Swift/Kotlin with interactions

---

## Limitations & Design Decisions

### Why CSS was excluded from Stage 1

The `ronantakizawa/webui` dataset includes separate `html` and `css` columns. CSS was excluded from Stage 1 training because:

- CSS inflates sequence length significantly — memory in VLM fine-tuning scales with the **square** of sequence length
- Stage 2 already handles styling via React + Tailwind
- Keeping Stage 1 focused on structure produced cleaner, more stable training

CSS (and animations) are the natural next step once the structural pipeline is validated.

### Why not implement interaction/animation conversion

Design tool internal JSON formats (layer trees, event graphs) are largely not publicly available. The `ronantakizawa/webui` dataset, however, does include animation-related data — making this a realistic near-term extension rather than a fundamental blocker.

---

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for a detailed log of issues encountered, including:

- `zsh: killed` — MLX per-process memory limit on Apple Silicon
- OOM with Llama 3.2 11B → switched to Qwen2-VL 7B
- Data format errors (JSONL → Parquet migration)
- RunPod disk space issues
- Cumulative memory leak during long training runs
