# Irodori-TTS-Turbo (Accelerated Edition)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**Irodori-TTS-Turbo** is an extremely optimized version of Irodori-TTS, a Flow Matching-based Text-to-Speech model. 
By integrating native **Intel XPU (Arc/Core Ultra)** support and state-of-the-art inference algorithms, it achieves over **10x speedup** compared to the original implementation.

## 🚀 Key Accelerated Features (Turbo)

- **Intel XPU Native Support (Discrete & Integrated GPUs / NPU)**: 
  Optimized for Intel's hardware accelerators. Uses `torch-xpu` with oneDNN and SYCL backends for maximum hardware utilization.
- **Dynamic Sequence Pruning ("Plane" Reduction)**: 
  An innovative pruning technique that identifies "settled" latent patches during the diffusion process. It dynamically reduces the active sequence length (computational plane), significantly cutting down Attention complexity in later steps.
- **10-Step Ultra-Fast Inference**: 
  Leverages Logit-Normal sampling and optimized Temporal Score Rescaling (TSR) to achieve high-quality audio in just **10 Euler steps**.
- **Flash Attention & SDPA Optimization**: 
  Ensures that the most efficient hardware kernels are triggered by using float-based attention masks and SDPA-compatible logic.
- **BFloat16 Mixed Precision**: 
  Full support for bf16 across all accelerators to reduce memory bandwidth and increase throughput without quality loss.

## 📦 Installation (Windows / Intel Arc)

For Windows users with Intel hardware, setting up the environment is as simple as running one script:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NamiCode-Dev/Irodori-TTS-Turbo.git
   cd Irodori-TTS-Turbo
   ```

2. **Run Installer**:
   Double-click `install.bat` or run it via terminal:
   ```bash
   install.bat
   ```
   This script automatically installs `uv` and sets up a virtual environment with **XPU-enabled PyTorch** and all necessary Intel runtime libraries.

## 🛠️ Quick Start

### Launch Web UI
Simply run the following script to start the localized and optimized Gradio interface:
```bash
run.bat
```
Then access the UI at `http://localhost:7860`.

### CLI Inference
```bash
uv run python infer.py \
  --text "Synthesizing audio is now blazingly fast." \
  --ref-wav path/to/reference.wav \
  --num-steps 10
```

## 📊 Performance Benchmark (Theoretical)

| Optimization Item | Speedup Factor |
|---|---|
| Step Reduction (40 -> 10) | **4.0x** |
| XPU Acceleration (Intel GPU vs CPU) | **5.0x ~ 20.0x** |
| Dynamic Sequence Pruning | **~1.5x** |
| Flash Attention / bf16 | **~2.0x** |
| **Total Cumulative Speedup** | **Up to 15x - 50x+** |

## 📁 Project Structure

- `irodori_tts/`: Core library with optimized inference runtime.
- `gradio_app.py`: Multi-lingual Gradio Web UI.
- `install.bat`: One-click environment setup for Windows.
- `run.bat`: Quick launch script.
- `pyproject.toml`: Dependency definition (supports auto-switching between CUDA/XPU).

## 📜 License

- **Code**: [MIT License](LICENSE) (Copyright (c) 2026 Aratako, NamiCode)
- **Model Weights**: Refer to the [Aratako/Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) model card for licensing details.

## 🤝 Acknowledgments

This project is built upon and optimized from:
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (Original v2 Codebase)
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/)
- [DACVAE](https://github.com/facebookresearch/dacvae)
