import os

def replace_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Title & Markdown
    content = content.replace('title="Irodori-TTS Gradio"', 'title="Irodori-TTS 音声合成 WebUI"')
    content = content.replace('title="Irodori-TTS VoiceDesign Gradio"', 'title="Irodori-TTS VoiceDesign WebUI"')
    content = content.replace('gr.Markdown("# Irodori-TTS Inference (Cached Runtime)")', 'gr.Markdown("# Irodori-TTS 音声合成 WebUI")')
    content = content.replace('gr.Markdown("# Irodori-TTS VoiceDesign Inference")', 'gr.Markdown("# Irodori-TTS VoiceDesign 音声合成 WebUI")')
    content = content.replace(
        '"When settings are unchanged, runtime is reused and only sampling/decoding runs."',
        '"モデルやデバイスの設定が同じ場合はメモリ上のモデルを再利用して高速に生成します。"'
    )
    content = content.replace(
        '"VoiceDesign版モデル向けのUIです。caption を入れると caption / style conditioning、空欄なら text-only conditioning で推論します。"',
        '"VoiceDesign版モデル向けのUIです。Caption（スタイルプロンプト）を入力するとそのスタイルで生成され、空欄の場合はテキストのみで推論します。"'
    )

    # Row 1
    content = content.replace('label="Checkpoint (.pt/.safetensors or HF repo id)"', 'label="モデル (チェックポイントのパス、またはHFリポジトリID)"')
    content = content.replace('label="Model Device"', 'label="推論デバイス (モデル)"')
    content = content.replace('label="Model Precision"', 'label="計算精度 (モデル)"')
    content = content.replace('label="Codec Device"', 'label="推論デバイス (コーデック)"')
    content = content.replace('label="Codec Precision"', 'label="計算精度 (コーデック)"')

    # Row 2
    content = content.replace('gr.Button("Load Model")', 'gr.Button("モデルを読み込む")')
    content = content.replace('gr.Button("Unload Model")', 'gr.Button("メモリから解放")')
    content = content.replace('label="Model Status"', 'label="モデルの状態 (ステータス)"')

    # Text & Audio
    content = content.replace('label="Text"', 'label="読み上げるテキスト"')
    content = content.replace('label="Caption / Style Prompt (optional)"', 'label="スタイルの指示 / キャプション (オプション)"')
    content = content.replace(
        'label="Reference Audio Upload (optional, blank = no-reference mode)"',
        'label="参照音声のアップロード (オプション、空欄の場合はノーリファレンスモードで生成)"'
    )

    # Sampling
    content = content.replace('gr.Accordion("Sampling", open=True)', 'gr.Accordion("サンプリング設定 (品質や生成速度の調整)", open=True)')
    content = content.replace('label="Num Steps"', 'label="ステップ数 (少ないほど高速、15程度で十分な品質)"')
    content = content.replace('label="Num Candidates"', 'label="生成候補数 (一度に生成する音声の数)"')
    content = content.replace('label="Seed (blank=random)"', 'label="シード値 (空欄でランダム生成)"')

    content = content.replace('label="CFG Guidance Mode"', 'label="CFG ガイダンスモード (independent推奨)"')
    content = content.replace('label="CFG Scale Text"', 'label="テキストの反映度 (CFG Scale Text)"')
    content = content.replace('label="CFG Scale Speaker"', 'label="話者の反映度 (CFG Scale Speaker)"')
    content = content.replace('label="CFG Scale Caption"', 'label="キャプションの反映度 (CFG Scale Caption)"')

    # Advanced
    content = content.replace('gr.Accordion("Advanced (Optional)", open=False)', 'gr.Accordion("高度な設定 (オプション)", open=False)')
    content = content.replace('label="CFG Scale Override (optional)"', 'label="CFG一括上書き (オプション)"')
    content = content.replace('label="CFG Min t"', 'label="CFG 最小t値 (CFGを適用する最小ステップ)"')
    content = content.replace('label="CFG Max t"', 'label="CFG 最大t値"')
    content = content.replace('label="Context KV Cache"', 'label="Context KVキャッシュを使用 (高速化)"')

    content = content.replace('label="Max Text Len (optional)"', 'label="テキストの最大トークン数 (オプション)"')
    content = content.replace('label="Max Caption Len (optional)"', 'label="キャプションの最大トークン数 (オプション)"')

    content = content.replace('label="Truncation Factor (optional)"', 'label="トランケーション係数 (オプション)"')
    content = content.replace('label="Rescale k (optional)"', 'label="Rescale k (スコア再スケーリング係数、0.7推奨)"')
    content = content.replace('label="Rescale sigma (optional)"', 'label="Rescale sigma (スコア再スケーリング分散、0.7推奨)"')

    content = content.replace('label="Speaker KV Scale (optional)"', 'label="話者特徴スケール (Speaker KV Scale、オプション)"')
    content = content.replace('label="Speaker KV Min t (optional)"', 'label="話者特徴スケール 最小t値"')
    content = content.replace('label="Speaker KV Max Layers (optional)"', 'label="話者特徴スケール 最大レイヤー数"')

    # Generate
    content = content.replace('gr.Button("Generate", variant="primary")', 'gr.Button("音声を生成 (Generate)", variant="primary")')

    # Outputs
    content = content.replace('label=f"Generated Audio {i + 1}"', 'label=f"生成された音声 {i + 1}"')
    content = content.replace('label="Run Log"', 'label="実行ログ"')
    content = content.replace('label="Timing"', 'label="処理時間 (タイミング)"')

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

replace_in_file("gradio_app.py")
replace_in_file("gradio_app_voicedesign.py")
print("UI translation complete.")
