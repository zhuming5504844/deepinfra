# deepinfra

本项目提供一个 Python GUI，用于调用 DeepInfra Whisper API 完成语音转录并生成字幕。

## 功能
- 支持拖放或选择音频文件
- 可配置 API Key / 模型 / 语言 / 提示词 / 温度等参数
- 支持请求超时与重试、字幕分段参数
- 内置运行日志
- 一键启动脚本（Windows: `start.bat`，macOS/Linux: `start.sh`）

## 环境准备
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## 一键启动
- macOS / Linux
  ```bash
  ./start.sh
  ```
- Windows
  ```bat
  start.bat
  ```

## 说明
默认使用本地 OpenAI 兼容接口：
`http://localhost:8000/v1/openai/audio/transcriptions`

也可选择 DeepInfra 远程接口：
- OpenAI 兼容：`https://api.deepinfra.com/v1/openai/audio/transcriptions`
- 推理接口：`https://api.deepinfra.com/v1/inference/openai/whisper-large-v3`

模型名称请使用 OpenAI 兼容格式（如 `openai/whisper-large-v3`）。如果只填写 `whisper-large-v3`，应用会自动补全为 `openai/whisper-large-v3`。

应用会将输出保存到所选目录，`verbose_json` 会额外生成 `.srt` 字幕文件。

应用会自动保存参数到用户目录下的 `.deepinfra_transcriber.json`，下次启动会自动恢复。

如遇 402 错误，表示账户余额不足，请在 DeepInfra 控制台充值或配置自动续费。
