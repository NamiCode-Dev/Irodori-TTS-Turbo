#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
    list_available_runtime_precisions,
    save_wav,
)

FIXED_SECONDS = 30.0
MAX_GRADIO_CANDIDATES = 32
GRADIO_AUDIO_COLS_PER_ROW = 4


def _default_checkpoint() -> str:
    candidates = sorted(
        [
            *Path(".").glob("**/checkpoint_*.pt"),
            *Path(".").glob("**/checkpoint_*.safetensors"),
        ]
    )
    if not candidates:
        return "Aratako/Irodori-TTS-500M-v2"
    return str(candidates[-1])


def _default_model_device() -> str:
    return default_runtime_device()


def _default_codec_device() -> str:
    return default_runtime_device()


def _precision_choices_for_device(device: str) -> list[str]:
    return list_available_runtime_precisions(device)


def _on_model_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _on_codec_device_change(device: str) -> gr.Dropdown:
    choices = _precision_choices_for_device(device)
    return gr.Dropdown(choices=choices, value=choices[0])


def _parse_optional_float(raw: str | None, label: str) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a float or blank.") from exc


def _parse_optional_int(raw: str | None, label: str) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an int or blank.") from exc


def _format_timings(stage_timings: list[tuple[str, float]], total_to_decode: float) -> str:
    lines = [
        "[timing] ---- request ----",
        *[f"[timing] {name}: {sec * 1000.0:.1f} ms" for name, sec in stage_timings],
        f"[timing] total_to_decode: {total_to_decode:.3f} s",
    ]
    return "\n".join(lines)


def _resolve_ref_wav(uploaded_audio: str | None) -> str | None:
    if uploaded_audio is not None and str(uploaded_audio).strip() != "":
        return str(uploaded_audio)
    return None


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint is required.")

    suffix = Path(checkpoint).suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        return checkpoint

    resolved = hf_hub_download(repo_id=checkpoint, filename="model.safetensors")
    print(f"[gradio] checkpoint: hf://{checkpoint} -> {resolved}", flush=True)
    return str(resolved)


def _build_runtime_key(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> RuntimeKey:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    # Auto-enable torch.compile only on CUDA (XPU Triton backend has issues on Windows).
    _should_compile = str(model_device).startswith("cuda")
    return RuntimeKey(
        checkpoint=checkpoint_path,
        model_device=str(model_device),
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision=str(model_precision),
        codec_device=str(codec_device),
        codec_precision=str(codec_precision),
        enable_watermark=bool(enable_watermark),
        compile_model=_should_compile,
        compile_dynamic=False,
    )


def _load_model(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
) -> str:
    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )
    _, reloaded = get_cached_runtime(runtime_key)
    if reloaded:
        status = "loaded model into memory"
    else:
        status = "model already loaded; reused existing runtime"
    return (
        f"{status}\n"
        f"checkpoint: {runtime_key.checkpoint}\n"
        f"model_device: {runtime_key.model_device}\n"
        f"model_precision: {runtime_key.model_precision}\n"
        f"codec_device: {runtime_key.codec_device}\n"
        f"codec_precision: {runtime_key.codec_precision}"
    )


def _run_generation(
    checkpoint: str,
    model_device: str,
    model_precision: str,
    codec_device: str,
    codec_precision: str,
    enable_watermark: bool,
    text: str,
    uploaded_audio: str | None,
    num_steps: int,
    num_candidates: int,
    seed_raw: str,
    cfg_guidance_mode: str,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_scale_raw: str,
    cfg_min_t: float,
    cfg_max_t: float,
    context_kv_cache: bool,
    truncation_factor_raw: str,
    rescale_k_raw: str,
    rescale_sigma_raw: str,
    speaker_kv_scale_raw: str,
    speaker_kv_min_t_raw: str,
    speaker_kv_max_layers_raw: str,
) -> tuple[object, ...]:
    def stdout_log(msg: str) -> None:
        print(msg, flush=True)

    runtime_key = _build_runtime_key(
        checkpoint=checkpoint,
        model_device=model_device,
        model_precision=model_precision,
        codec_device=codec_device,
        codec_precision=codec_precision,
        enable_watermark=enable_watermark,
    )

    if str(text).strip() == "":
        raise ValueError("text is required.")
    requested_candidates = int(num_candidates)
    if requested_candidates <= 0:
        raise ValueError("num_candidates must be >= 1.")
    if requested_candidates > MAX_GRADIO_CANDIDATES:
        raise ValueError(f"num_candidates must be <= {MAX_GRADIO_CANDIDATES}.")

    cfg_scale = _parse_optional_float(cfg_scale_raw, "cfg_scale")
    truncation_factor = _parse_optional_float(truncation_factor_raw, "truncation_factor")
    rescale_k = _parse_optional_float(rescale_k_raw, "rescale_k")
    rescale_sigma = _parse_optional_float(rescale_sigma_raw, "rescale_sigma")
    speaker_kv_scale = _parse_optional_float(speaker_kv_scale_raw, "speaker_kv_scale")
    speaker_kv_min_t = _parse_optional_float(speaker_kv_min_t_raw, "speaker_kv_min_t")
    speaker_kv_max_layers = _parse_optional_int(speaker_kv_max_layers_raw, "speaker_kv_max_layers")
    seed = _parse_optional_int(seed_raw, "seed")

    ref_wav = _resolve_ref_wav(uploaded_audio=uploaded_audio)
    no_ref = ref_wav is None
    ref_normalize_db = -16.0
    ref_ensure_max = True

    runtime, reloaded = get_cached_runtime(runtime_key)
    stdout_log(f"[gradio] runtime: {'reloaded' if reloaded else 'reused'}")
    stdout_log(
        (
            "[gradio] request: model_device={} model_precision={} codec_device={} codec_precision={} "
            "watermark={} mode={} seconds={} steps={} seed={} no_ref={} candidates={}"
        ).format(
            model_device,
            model_precision,
            codec_device,
            codec_precision,
            enable_watermark,
            cfg_guidance_mode,
            FIXED_SECONDS,
            num_steps,
            "random" if seed is None else seed,
            no_ref,
            requested_candidates,
        )
    )

    result = runtime.synthesize(
        SamplingRequest(
            text=str(text),
            ref_wav=ref_wav,
            ref_latent=None,
            no_ref=bool(no_ref),
            ref_normalize_db=ref_normalize_db,
            ref_ensure_max=bool(ref_ensure_max),
            num_candidates=requested_candidates,
            decode_mode="sequential",
            seconds=FIXED_SECONDS,
            max_ref_seconds=30.0,
            max_text_len=None,
            num_steps=int(num_steps),
            seed=None if seed is None else int(seed),
            cfg_guidance_mode=str(cfg_guidance_mode),
            cfg_scale_text=float(cfg_scale_text),
            cfg_scale_speaker=float(cfg_scale_speaker),
            cfg_scale=cfg_scale,
            cfg_min_t=float(cfg_min_t),
            cfg_max_t=float(cfg_max_t),
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            context_kv_cache=bool(context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            trim_tail=True,
        ),
        log_fn=stdout_log,
    )

    out_dir = Path("gradio_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_paths: list[str] = []
    for i, audio in enumerate(result.audios, start=1):
        out_path = save_wav(
            out_dir / f"sample_{stamp}_{i:03d}.wav",
            audio.float(),
            result.sample_rate,
        )
        out_paths.append(str(out_path))

    runtime_msg = "runtime: reloaded" if reloaded else "runtime: reused"
    detail_lines = [
        runtime_msg,
        f"seed_used: {result.used_seed}",
        f"candidates: {len(result.audios)}",
        *[f"saved[{i}]: {path}" for i, path in enumerate(out_paths, start=1)],
        *result.messages,
    ]
    detail_text = "\n".join(detail_lines)
    timing_text = _format_timings(result.stage_timings, result.total_to_decode)
    stdout_log(f"[gradio] saved {len(out_paths)} candidates")

    audio_updates: list[object] = []
    for i in range(MAX_GRADIO_CANDIDATES):
        if i < len(out_paths):
            audio_updates.append(gr.update(value=out_paths[i], visible=True))
        else:
            audio_updates.append(gr.update(value=None, visible=False))
    return (*audio_updates, detail_text, timing_text)


def _clear_runtime_cache() -> str:
    clear_cached_runtime()
    return "cleared loaded model from memory"


# Dictionary for bilingual support
TRANSLATIONS = {
    "English": {
        "title": "Irodori-TTS-Turbo WebUI",
        "description": "Accelerated TTS with Intel XPU & Dynamic Pruning. Reuses model in memory for fast generation.",
        "checkpoint": "Model (Path or HF Repo ID)",
        "model_device": "Inference Device (Model)",
        "model_precision": "Precision (Model)",
        "codec_device": "Inference Device (Codec)",
        "codec_precision": "Precision (Codec)",
        "load_model": "Load Model",
        "clear_cache": "Clear Memory",
        "status": "Model Status",
        "text": "Text to Synthesize",
        "ref_audio": "Reference Audio (Optional; Leave blank for no-reference mode)",
        "sampling_settings": "Sampling Settings (Quality & Speed)",
        "num_steps": "Number of Steps (Turbo: 10-15 steps recommended)",
        "num_candidates": "Number of Candidates",
        "seed": "Seed (Blank for random)",
        "cfg_mode": "CFG Guidance Mode (Independent recommended)",
        "cfg_text": "CFG Scale Text",
        "cfg_speaker": "CFG Scale Speaker",
        "advanced_settings": "Advanced Settings",
        "cfg_override": "CFG Override (Optional)",
        "cfg_min_t": "CFG Min t",
        "cfg_max_t": "CFG Max t",
        "use_kv_cache": "Use Context KV Cache (Faster)",
        "truncation": "Truncation Factor (Optional)",
        "rescale_k": "Rescale k (Recommended: 0.7)",
        "rescale_sigma": "Rescale sigma (Recommended: 0.7)",
        "speaker_kv_scale": "Speaker KV Scale (Optional)",
        "speaker_kv_min_t": "Speaker KV Min t",
        "speaker_kv_max_layers": "Speaker KV Max Layers",
        "generate": "Generate Audio",
        "log": "Execution Log",
        "timing": "Processing Time (Timings)",
        "output_label": "Generated Audio",
        "model_tab": "Model Configuration",
        "sampling_tab": "Sampling",
        "advanced_tab": "Advanced",
        "log_tab": "Logs & Info",
        "emojis": "Emojis",
    },
    "日本語": {
        "title": "Irodori-TTS-Turbo 音声合成 WebUI",
        "description": "Intel XPU対応・動的削減による高速化版。メモリ上のモデルを再利用して高速に生成します。",
        "checkpoint": "モデル (パスまたはHFリポジトリID)",
        "model_device": "推論デバイス (モデル)",
        "model_precision": "計算精度 (モデル)",
        "codec_device": "推論デバイス (コーデック)",
        "codec_precision": "計算精度 (コーデック)",
        "load_model": "モデルを読み込む",
        "clear_cache": "メモリから解放",
        "status": "モデルの状態 (ステータス)",
        "text": "読み上げるテキスト",
        "ref_audio": "参照音声のアップロード (オプション、空欄の場合はノーリファレンスモード)",
        "sampling_settings": "サンプリング設定 (品質や生成速度の調整)",
        "num_steps": "ステップ数 (Turbo版は10-15程度推奨)",
        "num_candidates": "生成候補数",
        "seed": "シード値 (空欄でランダム)",
        "cfg_mode": "CFG ガイダンスモード (independent推奨)",
        "cfg_text": "テキストの反映度 (CFG Scale Text)",
        "cfg_speaker": "話者の反映度 (CFG Scale Speaker)",
        "advanced_settings": "高度な設定",
        "cfg_override": "CFG一括上書き (オプション)",
        "cfg_min_t": "CFG 最小t値",
        "cfg_max_t": "CFG 最大t値",
        "use_kv_cache": "Context KVキャッシュを使用 (高速化)",
        "truncation": "トランケーション係数 (オプション)",
        "rescale_k": "Rescale k (推奨: 0.7)",
        "rescale_sigma": "Rescale sigma (推奨: 0.7)",
        "speaker_kv_scale": "話者特徴スケール (オプション)",
        "speaker_kv_min_t": "話者特徴スケール 最小t値",
        "speaker_kv_max_layers": "話者特徴スケール 最大レイヤー数",
        "generate": "音声を生成",
        "log": "実行ログ",
        "timing": "処理時間 (タイミング)",
        "output_label": "生成された音声",
        "model_tab": "モデル設定",
        "sampling_tab": "生成設定",
        "advanced_tab": "詳細設定",
        "log_tab": "ログと情報",
        "emojis": "絵文字一覧",
    }
}

def _update_ui_language(lang: str):
    t = TRANSLATIONS[lang]
    updates = [
        gr.update(value=f"# {t['title']}"),
        gr.update(value=t['description']),
        gr.update(label=t['checkpoint']),
        gr.update(label=t['model_device']),
        gr.update(label=t['model_precision']),
        gr.update(label=t['codec_device']),
        gr.update(label=t['codec_precision']),
        gr.update(value=t['load_model']),
        gr.update(value=t['clear_cache']),
        gr.update(label=t['status']),
        gr.update(label=t['text']),
        gr.update(label=t['ref_audio']),
        gr.update(label=t['num_steps']),
        gr.update(label=t['num_candidates']),
        gr.update(label=t['seed']),
        gr.update(label=t['cfg_mode']),
        gr.update(label=t['cfg_text']),
        gr.update(label=t['cfg_speaker']),
        gr.update(label=t['cfg_override']),
        gr.update(label=t['cfg_min_t']),
        gr.update(label=t['cfg_max_t']),
        gr.update(label=t['use_kv_cache']),
        gr.update(label=t['truncation']),
        gr.update(label=t['rescale_k']),
        gr.update(label=t['rescale_sigma']),
        gr.update(label=t['speaker_kv_scale']),
        gr.update(label=t['speaker_kv_min_t']),
        gr.update(label=t['speaker_kv_max_layers']),
        gr.update(value=t['generate']),
        gr.update(label=t['log']),
        gr.update(label=t['timing']),
        gr.update(label=t['model_tab']),
        gr.update(label=t['sampling_tab']),
        gr.update(label=t['advanced_tab']),
        gr.update(label=t['log_tab']),
        gr.update(label=t['emojis']),
    ]
    # Update audio output labels
    for i in range(MAX_GRADIO_CANDIDATES):
        updates.append(gr.update(label=f"{t['output_label']} {i + 1}"))
    return updates



def build_ui() -> gr.Blocks:
    default_checkpoint = _default_checkpoint()
    default_model_device = _default_model_device()
    default_codec_device = _default_codec_device()
    device_choices = list_available_runtime_devices()
    model_precision_choices = _precision_choices_for_device(default_model_device)
    codec_precision_choices = _precision_choices_for_device(default_codec_device)

    # Initial language: English
    t = TRANSLATIONS["English"]

    with gr.Blocks(title=t["title"]) as demo:
        with gr.Row():
            with gr.Column(scale=4):
                title_md = gr.Markdown(f"# {t['title']}")
                description_md = gr.Markdown(t["description"])
            with gr.Column(scale=1):
                language_select = gr.Radio(
                    choices=["English", "日本語"],
                    value="English",
                    label="Language / 言語",
                )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    text = gr.Textbox(
                        label=t["text"],
                        lines=8,
                        placeholder="Enter text to synthesize...",
                        elem_id="input-text"
                    )
                    
                    with gr.Accordion(t["emojis"], open=False) as emoji_accordion:
                        with gr.Group():
                            emojis = [
                                "👂", "😮‍💨", "⏸️", "🤭", "🥵", "📢", "😏", "🥺", "🌬️", "😮",
                                "👅", "💋", "🫶", "😭", "😱", "😪", "⏩", "📞", "🐢", "🥤",
                                "🤧", "😒", "😰", "😆", "😠", "😲", "🥱", "😖", "😟", "🫣",
                                "🙄", "😊", "👌", "🙏", "🥴", "🎵", "🤐", "😌", "🤔"
                            ]
                            for i in range(0, len(emojis), 10):
                                with gr.Row():
                                    for emoji in emojis[i : i + 10]:
                                        btn = gr.Button(emoji, min_width=42, variant="secondary")
                                        btn.click(
                                            fn=lambda x: x,
                                            inputs=[text],
                                            outputs=[text],
                                            js=f"""(text_val) => {{
                                                const textarea = document.querySelector('#input-text textarea');
                                                const emoji = "{emoji}";
                                                if (!textarea) return (text_val || "") + emoji;
                                                const start = textarea.selectionStart;
                                                const end = textarea.selectionEnd;
                                                const newVal = text_val.slice(0, start) + emoji + text_val.slice(end);
                                                textarea.value = newVal;
                                                textarea.setSelectionRange(start + emoji.length, start + emoji.length);
                                                textarea.dispatchEvent(new Event('input'));
                                                return newVal;
                                            }}"""
                                        )

                    uploaded_audio = gr.Audio(
                        label=t["ref_audio"],
                        type="filepath",
                    )
                    generate_btn = gr.Button(t["generate"], variant="primary", size="lg")

                with gr.Group():
                    out_audios: list[gr.Audio] = []
                    num_rows = (
                        MAX_GRADIO_CANDIDATES + GRADIO_AUDIO_COLS_PER_ROW - 1
                    ) // GRADIO_AUDIO_COLS_PER_ROW
                    for row_idx in range(num_rows):
                        with gr.Row():
                            for col_idx in range(GRADIO_AUDIO_COLS_PER_ROW):
                                i = row_idx * GRADIO_AUDIO_COLS_PER_ROW + col_idx
                                if i >= MAX_GRADIO_CANDIDATES:
                                    break
                                out_audios.append(
                                    gr.Audio(
                                        label=f"{t['output_label']} {i + 1}",
                                        type="filepath",
                                        interactive=False,
                                        visible=(i == 0),
                                        min_width=160,
                                    )
                                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab(t["sampling_tab"]) as sampling_tab:
                        with gr.Group():
                            num_steps = gr.Slider(label=t["num_steps"], minimum=1, maximum=120, value=15, step=1)
                            num_candidates = gr.Slider(
                                label=t["num_candidates"],
                                minimum=1,
                                maximum=MAX_GRADIO_CANDIDATES,
                                value=1,
                                step=1,
                            )
                            seed_raw = gr.Textbox(label=t["seed"], value="")

                        with gr.Group():
                            cfg_guidance_mode = gr.Dropdown(
                                label=t["cfg_mode"],
                                choices=["independent", "joint", "alternating"],
                                value="independent",
                            )
                            cfg_scale_text = gr.Slider(
                                label=t["cfg_text"],
                                minimum=0.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.1,
                            )
                            cfg_scale_speaker = gr.Slider(
                                label=t["cfg_speaker"],
                                minimum=0.0,
                                maximum=10.0,
                                value=5.0,
                                step=0.1,
                            )

                    with gr.Tab(t["model_tab"]) as model_tab:
                        checkpoint = gr.Textbox(
                            label=t["checkpoint"],
                            value=default_checkpoint,
                        )
                        with gr.Row():
                            model_device = gr.Dropdown(
                                label=t["model_device"],
                                choices=device_choices,
                                value=default_model_device,
                            )
                            model_precision = gr.Dropdown(
                                label=t["model_precision"],
                                choices=model_precision_choices,
                                value=model_precision_choices[0],
                            )
                        with gr.Row():
                            codec_device = gr.Dropdown(
                                label=t["codec_device"],
                                choices=device_choices,
                                value=default_codec_device,
                            )
                            codec_precision = gr.Dropdown(
                                label=t["codec_precision"],
                                choices=codec_precision_choices,
                                value=codec_precision_choices[0],
                            )
                        enable_watermark = gr.State(False)
                        
                        with gr.Row():
                            load_model_btn = gr.Button(t["load_model"])
                            clear_cache_btn = gr.Button(t["clear_cache"])
                        clear_cache_msg = gr.Textbox(label=t["status"], interactive=False)

                    with gr.Tab(t["advanced_tab"]) as advanced_tab:
                        cfg_scale_raw = gr.Textbox(label=t["cfg_override"], value="")
                        with gr.Row():
                            cfg_min_t = gr.Number(label=t["cfg_min_t"], value=0.5)
                            cfg_max_t = gr.Number(label=t["cfg_max_t"], value=1.0)
                        context_kv_cache = gr.Checkbox(label=t["use_kv_cache"], value=True)
                        with gr.Row():
                            truncation_factor_raw = gr.Textbox(label=t["truncation"], value="")
                            rescale_k_raw = gr.Textbox(label=t["rescale_k"], value="0.7")
                        rescale_sigma_raw = gr.Textbox(label=t["rescale_sigma"], value="0.7")
                        with gr.Row():
                            speaker_kv_scale_raw = gr.Textbox(label=t["speaker_kv_scale"], value="")
                            speaker_kv_min_t_raw = gr.Textbox(label=t["speaker_kv_min_t"], value="0.9")
                        speaker_kv_max_layers_raw = gr.Textbox(
                            label=t["speaker_kv_max_layers"], value=""
                        )

                    with gr.Tab(t["log_tab"]) as log_tab:
                        out_log = gr.Textbox(label=t["log"], lines=12)
                        out_timing = gr.Textbox(label=t["timing"], lines=12)

        # Language Switch Event
        language_select.change(
            _update_ui_language,
            inputs=[language_select],
            outputs=[
                title_md, description_md, checkpoint, model_device, model_precision,
                codec_device, codec_precision, load_model_btn, clear_cache_btn,
                clear_cache_msg, text, uploaded_audio, num_steps,
                num_candidates, seed_raw, cfg_guidance_mode, cfg_scale_text,
                cfg_scale_speaker, cfg_scale_raw, cfg_min_t,
                cfg_max_t, context_kv_cache, truncation_factor_raw, rescale_k_raw,
                rescale_sigma_raw, speaker_kv_scale_raw, speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw, generate_btn, out_log, out_timing,
                model_tab, sampling_tab, advanced_tab, log_tab, emoji_accordion,
                *out_audios
            ]
        )

        generate_btn.click(
            _run_generation,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
                text,
                uploaded_audio,
                num_steps,
                num_candidates,
                seed_raw,
                cfg_guidance_mode,
                cfg_scale_text,
                cfg_scale_speaker,
                cfg_scale_raw,
                cfg_min_t,
                cfg_max_t,
                context_kv_cache,
                truncation_factor_raw,
                rescale_k_raw,
                rescale_sigma_raw,
                speaker_kv_scale_raw,
                speaker_kv_min_t_raw,
                speaker_kv_max_layers_raw,
            ],
            outputs=[*out_audios, out_log, out_timing],
        )
        model_device.change(
            _on_model_device_change, inputs=[model_device], outputs=[model_precision]
        )
        codec_device.change(
            _on_codec_device_change, inputs=[codec_device], outputs=[codec_precision]
        )

        load_model_btn.click(
            _load_model,
            inputs=[
                checkpoint,
                model_device,
                model_precision,
                codec_device,
                codec_precision,
                enable_watermark,
            ],
            outputs=[clear_cache_msg],
        )
        clear_cache_btn.click(_clear_runtime_cache, outputs=[clear_cache_msg])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio app for Irodori-TTS with cached runtime.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    
    print(f"\n[Irodori-TTS-Turbo] Server starting...")
    print(f" - Local Access:   http://localhost:{args.server_port}")
    print(f" - Network Access: http://0.0.0.0:{args.server_port}\n")
    
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=bool(args.share),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()

