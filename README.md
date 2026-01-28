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
默认使用 DeepInfra OpenAI 兼容接口：
`https://api.deepinfra.com/v1/openai/audio/transcriptions`

应用会将输出保存到所选目录，`verbose_json` 会额外生成 `.srt` 字幕文件。
