# deepinfra

本项目提供一个 Python GUI，用于调用 DeepInfra Whisper API 完成语音转录并生成字幕。

## 功能
- 支持拖放或选择音频文件（支持常见音频格式）
- 支持批量队列：添加文件 / 移除选中 / 清空队列
- 可配置 API Key / API 地址 / 模型 / 输出粒度
- 默认保存常用配置到本地设置文件
- 内置运行日志
- 一键启动脚本（Windows: `start.bat`，macOS/Linux: `start.sh`）
- 预转录/异步转录前处理链路新增轮询间隔与整体超时控制，避免无限等待

## 目录结构（便于打包 EXE）
```text
.
├─ gui.py
├─ pyproject.toml
├─ requirements.txt
├─ src/
│  └─ deepinfra_transcriber/
│     ├─ __init__.py
│     └─ gui_app.py
├─ start.bat
└─ start.sh
```

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

## 打包 EXE（PyInstaller）
```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --windowed --name deepinfra-gui gui.py
```
打包完成后，产物在 `dist/deepinfra-gui.exe`。

## 说明
默认使用 DeepInfra Inference 接口：
`https://api.deepinfra.com/v1/inference/openai/whisper-large-v3`

模型名称请使用 DeepInfra 版本（如 `openai/whisper-large-v3`）。如果只填写 `whisper-large-v3`，应用会自动补全为 `openai/whisper-large-v3`。

应用会将输出保存为 `.srt` 字幕文件。

如遇 402 错误，表示账户余额不足，请在 DeepInfra 控制台充值或配置自动续费。
