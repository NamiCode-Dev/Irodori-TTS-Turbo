[English](#irodori-tts-turbo-accelerated-edition) | [日本語](#irodori-tts-turbo-加速版)

---

# Irodori-TTS-Turbo (Accelerated Edition)

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-500M-v2)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

**Irodori-TTS-Turbo** is an extremely optimized version of Irodori-TTS, a Flow Matching-based Text-to-Speech model. 
By integrating native **Intel XPU (Arc/Core Ultra)** support and state-of-the-art inference algorithms, it achieves over **10x speedup** on Intel hardware compared to the original CPU-based or unoptimized implementations. (Other accelerators like NVIDIA CUDA also see significant 4-6x gains from algorithmic optimizations).

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

## ✨ User-Friendly Features

- **Zero-Config Installer (`install.bat`)**: 
  Automated environment setup using `uv`. No need to manually install Python or PyTorch; the script handles everything including Intel-specific drivers and libraries.
- **One-Click Launcher (`run.bat`)**: 
  Instantly starts the Gradio Web UI without touching the terminal.
- **Bilingual Interface**: 
  Fully localized Web UI (English/Japanese) with dynamic language switching.
- **Smart Hardware Detection**: 
  Automatically selects and optimizes for the best available hardware (NVIDIA CUDA, Intel XPU, Apple MPS, or CPU).

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

---

# Irodori-TTS-Turbo (加速版)

**Irodori-TTS-Turbo** は、Flow Matchingベースの音声合成モデル Irodori-TTS を極限まで高速化した最適化版です。
特に **Intel XPU (Arc/Core Ultra)** へのネイティブ対応と、最新の推論最適化アルゴリズムにより、Intel環境では従来（CPU実行等）比で **10倍以上の高速化** を実現しています。（NVIDIA CUDA等でも、アルゴリズム改善により4〜6倍程度の高速化が期待できます）。

## 🚀 高速化機能 (Turbo)

- **Intel XPU ネイティブ対応 (GPU / NPU)**: 
  Intel のディスクリートGPU、内蔵GPU、および NPU に最適化。`torch-xpu` と oneDNN/SYCL バックエンドを利用し、ハードウェアの性能を最大限に引き出します。
- **動的平面削減 (Dynamic Sequence Pruning)**: 
  推論途中で「変化が完了したパッチ」を特定し計算対象から除外することで、Attentionの計算量を動的に削減する革新的な手法を導入しています。
- **10ステップ爆速推論**: 
  Logit-Normal サンプリングと最適化された TSR により、わずか **10ステップ** で高品質な音声を生成可能（標準の 40 ステップから大幅短縮）。
- **Flash Attention / SDPA 最適化**: 
  浮動小数点ベースのアテンションマスクを採用し、ハードウェアが提供する最速の計算パス（SDPA）を確実に利用します。
- **BFloat16 混合精度推論**: 
  メモリ帯域を節約し、精度を維持したままスループットを向上させます。

## ✨ 使いやすさへのこだわり

- **ゼロ構成インストーラー (`install.bat`)**: 
  `uv` を活用した全自動環境構築。Python や PyTorch の手動インストールは不要で、Intel 固有のドライバやライブラリまで一括でセットアップします。
- **ワンクリック起動 (`run.bat`)**: 
  ターミナルを意識することなく、ダブルクリックだけで Gradio Web UI を起動可能。
- **完全日本語対応 UI**: 
  日本語と英語のバイリンガルインターフェースを搭載し、設定から操作までスムーズに行えます。
- **自動ハードウェア検知**: 
  実行環境を自動判別し、NVIDIA CUDA、Intel XPU、Apple MPS、または CPU から最適なデバイスを選択し最適化します。

## 📦 インストール (Windows)

Windows ユーザー、特に Intel ハードウェアをお使いの場合は、スクリプト一つで環境構築が完了します。

1. **リポジトリをクローン**:
   ```bash
   git clone https://github.com/NamiCode-Dev/Irodori-TTS-Turbo.git
   cd Irodori-TTS-Turbo
   ```

2. **インストーラーを実行**:
   `install.bat` を実行してください。
   `uv` のインストールから、**Intel XPU 対応版 PyTorch** のセットアップまで自動で行われます。

## 🛠️ クイックスタート

### Web UI 起動
`run.bat` を実行し、ブラウザで `http://localhost:7860` を開いてください。最適化された日本語インターフェースが立ち上がります。

### CLI 推論
```bash
uv run python infer.py \
  --text "驚くほど速くなりましたね。" \
  --ref-wav path/to/reference.wav \
  --num-steps 10
```

## 📜 License

- **Code**: [MIT License](LICENSE) (Copyright (c) 2026 Aratako, NamiCode)
- **Model Weights**: Refer to the [Aratako/Irodori-TTS-500M-v2](https://huggingface.co/Aratako/Irodori-TTS-500M-v2) model card for licensing details.

## 🤝 Acknowledgments

This project is built upon and optimized from:
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (Original v2 Codebase)
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/)
- [DACVAE](https://github.com/facebookresearch/dacvae)
