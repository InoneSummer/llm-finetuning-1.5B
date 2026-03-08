# Project A — UI Screenshot → HTML (Stage 1)

## Motivation

No-code and low-code tools are becoming increasingly important in modern software development. Tools like Figma, Webflow, and ProtoPie allow designers to create UI prototypes — but the gap between a visual design and production-ready code still requires significant manual effort from developers.

The core question this project explores:

> **Can a vision-language model learn to look at a UI screenshot and generate the corresponding HTML structure?**

This is Stage 1 of a 2-stage pipeline. The goal is not to generate pixel-perfect code, but to extract the semantic HTML structure from a visual input — laying the foundation for a no-code tool that converts designs directly into code.

---

## What This Project Does

Fine-tunes **Qwen2-VL-7B** on (screenshot, HTML) pairs so that the model can generate HTML structure from a UI image.

- **Input**: UI screenshot
- **Output**: HTML code (structure only, no CSS)

### Why HTML only — not HTML + CSS?

The dataset (`ronantakizawa/webui`) includes separate `html`, `css`, and `js` columns. CSS was intentionally excluded for two reasons:

1. **Sequence length** — CSS inflates token count significantly. Memory usage in VLM fine-tuning scales with the square of sequence length (2× length = 4× memory). Including CSS would require far more compute.
2. **Role separation** — In the full 2-stage pipeline, Stage 2 handles styling via React + Tailwind. Stage 1 focuses purely on structural understanding.

---

## Model

| Item | Detail |
|------|--------|
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Quantization | 4-bit |
| Method | LoRA (rank=16, alpha=32) |
| Framework | Unsloth + HuggingFace TRL |
| Hardware | RunPod A100 80GB PCIe |
| Training time | ~32 minutes |
| Final loss | 0.7341 |

### Why Qwen2-VL?

- Rated as one of the top open-source vision-language models for web UI understanding
- 7B parameters fits comfortably within GPU memory constraints
- Strong performance on layout analysis and code generation tasks
- Originally attempted with Llama 3.2 11B on Apple Silicon (Mac Mini 64GB) — consistently crashed with OOM. Qwen2-VL 7B resolved this entirely.

### Why Unsloth?

Unsloth provides custom CUDA kernels optimized specifically for LoRA fine-tuning:
- ~2x faster training speed
- ~60% memory reduction vs standard PEFT
- Enabled stable training on a single A100 that would otherwise require multi-GPU setup

---

## Dataset

| Item | Detail |
|------|--------|
| Source | `ronantakizawa/webui` (HuggingFace) |
| Total samples | ~29,000 (train) |
| Filtered | HTML ≤ 4,000 characters |
| Used for training | 2,000 samples |
| Validation | 200 samples |

### Why filtering?

The dataset contains real-world production web pages. Some HTML samples exceed 400,000 characters. A single outlier sample is enough to cause an OOM crash during training.

```
Original:  ~29,000 samples (max HTML: 434,140 chars)
Filtered:   15,174 samples (≤ 4,000 chars)
Used:        2,000 samples
```

---

## Results

### Before Fine-tuning (Base Model)

The base model generates generic Tailwind-based HTML with no relationship to the actual screenshot:

```html
<html>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<body class="bg-gray-100 font-sans leading-normal tracking-normal">
    <div class="flex flex-col min-h-screen">
        <header class="bg-gray-800 text-white p-4">
```

### After Fine-tuning

The fine-tuned model generates structured HTML that matches the actual document structure:

```html
<!DOCTYPE html>
<html lang="en">
 <head>
  <meta charset="utf-8"/>
  <meta content="width=device-width, initial-scale=1.0" name="viewport"/>
  <title>Markdown Editor</title>
  <link href="https://fonts.googleapis.com" rel="preconnect"/>
```

Key improvements:
- Correct `<!DOCTYPE html>` declaration
- Proper `lang` attribute
- Accurate `<title>` tag matching the actual page
- Semantic meta tags and document structure

---

## Project Structure

```
project-a-vision/
├── get_data.py          # Download and explore dataset
├── prepare_data.py      # Initial data preprocessing
├── convert_data.py      # Format conversion
├── truncate_data.py     # HTML length filtering
├── train.py             # Fine-tuning script (Unsloth + TRL)
├── inference.py         # Before/after comparison script
├── adapters_qwen/       # Trained LoRA adapters
└── example/             # Sample inputs and outputs
```

---

## Training Command

```bash
python train.py
```

Full training configuration in `train.py`:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `learning_rate=1e-5`
- `num_train_epochs=1`
- `optim="adamw_8bit"`
- `lr_scheduler_type="cosine"`

---

## Troubleshooting

See [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) for a detailed log of issues encountered during development, including:

- Data format errors (`messages`, `images` column not found)
- OOM crashes on Apple Silicon with Llama 3.2 11B
- Migration from MLX (Apple Silicon) to Unsloth (CUDA)
- Disk space issues on RunPod
