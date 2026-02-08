import json
import math
import os
import textwrap
import time
from dataclasses import dataclass
from typing import List, Optional

import pysrt
import requests
from PyQt5 import QtCore, QtGui, QtWidgets

API_URL = "https://api.deepinfra.com/v1/openai/audio/transcriptions"


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionOptions:
    api_key: str
    model: str
    language: str
    prompt: str
    temperature: float
    response_format: str
    timestamp_granularities: List[str]
    timeout_seconds: int
    max_retries: int
    max_chars_per_segment: int
    max_segment_duration: float
    max_silence_duration: float
    audio_path: str
    output_dir: str


class TranscriptionWorker(QtCore.QObject):
    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, options: TranscriptionOptions) -> None:
        super().__init__()
        self.options = options

    def run(self) -> None:
        try:
            output_path = self._transcribe(self.options)
            self.finished.emit(output_path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _transcribe(self, options: TranscriptionOptions) -> str:
        os.makedirs(options.output_dir, exist_ok=True)

        headers = {"Authorization": f"Bearer {options.api_key}"} if options.api_key else {}

        model_name = self._normalize_model(options.model)
        if model_name != options.model:
            self.log_message.emit(f"模型已自动补全为: {model_name}")

        data = {
            "model": model_name,
            "language": options.language or None,
            "prompt": options.prompt or None,
            "temperature": options.temperature,
            "response_format": options.response_format,
        }
        if options.timestamp_granularities:
            data["timestamp_granularities"] = json.dumps(options.timestamp_granularities)

        payload = {key: value for key, value in data.items() if value is not None}

        filename = os.path.splitext(os.path.basename(options.audio_path))[0]
        output_extension = "srt" if options.response_format in {"srt", "vtt"} else "txt"
        output_path = os.path.join(options.output_dir, f"{filename}.{output_extension}")

        for attempt in range(options.max_retries + 1):
            try:
                with open(options.audio_path, "rb") as audio_file:
                    files = {"file": (os.path.basename(options.audio_path), audio_file)}
                    self.log_message.emit("正在请求 API...")
                    response = requests.post(
                        API_URL,
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=options.timeout_seconds,
                    )
                if response.status_code >= 400:
                    raise RuntimeError(self._format_api_error(response))

                if options.response_format in {"srt", "vtt"}:
                    output_text = response.text
                    if options.response_format == "srt":
                        output_text = self._split_srt_text(
                            output_text,
                            options.max_chars_per_segment,
                            options.max_segment_duration,
                        )
                    with open(output_path, "w", encoding="utf-8") as output_file:
                        output_file.write(output_text)
                    return output_path

                if options.response_format == "verbose_json":
                    result = response.json()
                    srt = self._build_srt_from_segments(
                        result.get("segments", []),
                        options.max_chars_per_segment,
                        options.max_segment_duration,
                        options.max_silence_duration,
                    )
                    output_path = os.path.join(options.output_dir, f"{filename}.srt")
                    with open(output_path, "w", encoding="utf-8") as output_file:
                        output_file.write(srt)
                    return output_path

                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(response.text)
                return output_path
            except Exception as exc:  # noqa: BLE001
                if attempt >= options.max_retries:
                    raise exc
                wait_time = 2 ** attempt
                self.log_message.emit(f"请求失败，{wait_time} 秒后重试: {exc}")
                time.sleep(wait_time)

        raise RuntimeError("无法完成转录，请检查网络或 API Key。")

    def _build_srt_from_segments(
        self,
        segments_data: List[dict],
        max_chars: int,
        max_duration: float,
        max_silence: float,
    ) -> str:
        segments: List[Segment] = []
        for segment in segments_data:
            text = (segment.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                Segment(
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    text=text,
                )
            )

        merged: List[Segment] = []
        current: Optional[Segment] = None

        for seg in segments:
            if current is None:
                current = Segment(seg.start, seg.end, seg.text)
                continue

            gap = seg.start - current.end
            combined_text = f"{current.text} {seg.text}".strip()
            duration = seg.end - current.start

            if (
                len(combined_text) <= max_chars
                and duration <= max_duration
                and gap <= max_silence
            ):
                current.end = seg.end
                current.text = combined_text
            else:
                merged.append(current)
                current = Segment(seg.start, seg.end, seg.text)

        if current:
            merged.append(current)

        refined: List[Segment] = []
        for seg in merged:
            refined.extend(self._split_segment(seg, max_chars, max_duration))

        srt = pysrt.SubRipFile()
        for idx, seg in enumerate(refined, start=1):
            srt.append(
                pysrt.SubRipItem(
                    index=idx,
                    start=pysrt.SubRipTime(milliseconds=int(seg.start * 1000)),
                    end=pysrt.SubRipTime(milliseconds=int(seg.end * 1000)),
                    text=seg.text,
                )
            )

        return srt.to_string()

    def _split_srt_text(self, srt_text: str, max_chars: int, max_duration: float) -> str:
        try:
            items = pysrt.from_string(srt_text)
        except Exception:  # noqa: BLE001
            return srt_text

        refined: List[Segment] = []
        for item in items:
            start_seconds = item.start.ordinal / 1000.0
            end_seconds = item.end.ordinal / 1000.0
            refined.extend(
                self._split_segment(
                    Segment(start=start_seconds, end=end_seconds, text=item.text),
                    max_chars,
                    max_duration,
                )
            )

        srt = pysrt.SubRipFile()
        for idx, seg in enumerate(refined, start=1):
            srt.append(
                pysrt.SubRipItem(
                    index=idx,
                    start=pysrt.SubRipTime(milliseconds=int(seg.start * 1000)),
                    end=pysrt.SubRipTime(milliseconds=int(seg.end * 1000)),
                    text=seg.text,
                )
            )

        return srt.to_string()

    def _split_segment(self, segment: Segment, max_chars: int, max_duration: float) -> List[Segment]:
        text = segment.text.strip()
        if not text:
            return []

        if max_chars <= 0:
            chunks = [text]
        else:
            chunks = textwrap.wrap(
                text,
                width=max_chars,
                break_long_words=True,
                break_on_hyphens=False,
            )

        total_duration = max(segment.end - segment.start, 0.0)
        if max_duration > 0 and total_duration > 0:
            required_count = max(len(chunks), math.ceil(total_duration / max_duration))
            if required_count > len(chunks):
                target_width = max(1, math.ceil(len(text) / required_count))
                if max_chars > 0:
                    target_width = min(max_chars, target_width)
                chunks = textwrap.wrap(
                    text,
                    width=target_width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            if len(chunks) < required_count:
                chunks = self._ensure_chunk_count(chunks, required_count)

        total_chars = sum(len(chunk) for chunk in chunks) or 1
        current_start = segment.start
        refined = []
        for index, chunk in enumerate(chunks):
            if index == len(chunks) - 1:
                chunk_end = segment.end
            else:
                chunk_duration = total_duration * len(chunk) / total_chars
                chunk_end = current_start + chunk_duration
            refined.append(Segment(current_start, chunk_end, chunk))
            current_start = chunk_end
        return refined

    def _ensure_chunk_count(self, chunks: List[str], target_count: int) -> List[str]:
        normalized = [chunk.strip() for chunk in chunks if chunk.strip()]
        while len(normalized) < target_count:
            longest_index = max(range(len(normalized)), key=lambda i: len(normalized[i]))
            text = normalized[longest_index]
            if len(text) <= 1:
                break
            mid = len(text) // 2
            split_index = text.rfind(" ", 0, mid)
            if split_index == -1:
                split_index = text.find(" ", mid)
            if split_index == -1:
                first, second = text[:mid], text[mid:]
            else:
                first, second = text[:split_index], text[split_index + 1 :]
            normalized[longest_index : longest_index + 1] = [
                part.strip() for part in (first, second) if part.strip()
            ]
        return normalized

    def _format_api_error(self, response: requests.Response) -> str:
        status = response.status_code
        body = response.text.strip()
        if status == 402:
            return "API 返回错误 402: 账户余额不足，请在 DeepInfra 控制台充值或配置自动续费。"
        if status == 401:
            return "API 返回错误 401: API Key 无效或缺失。"
        if status == 404:
            return "API 返回错误 404: 模型不存在，请确认模型名称（如 openai/whisper-large-v3）。"
        return f"API 返回错误 {status}: {body}"

    def _normalize_model(self, model: str) -> str:
        trimmed = model.strip()
        if not trimmed:
            return "openai/whisper-large-v3"
        if "/" not in trimmed:
            return f"openai/{trimmed}"
        return trimmed


class TranscriptionWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepInfra 语音转录字幕")
        self.setMinimumSize(900, 760)
        self.resize(980, 820)
        self.setAcceptDrops(True)

        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[TranscriptionWorker] = None

        self._build_ui()
        self._apply_theme()
        self._log("应用启动完成。拖放音频文件到窗口即可自动填入路径。")

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.file_group = QtWidgets.QGroupBox("音频文件")
        file_layout = QtWidgets.QHBoxLayout(self.file_group)
        self.audio_path_edit = QtWidgets.QLineEdit()
        self.audio_path_edit.setPlaceholderText("选择或拖放音频文件...")
        file_layout.addWidget(self.audio_path_edit)

        browse_button = QtWidgets.QPushButton("选择文件")
        browse_button.clicked.connect(self._choose_file)
        file_layout.addWidget(browse_button)

        main_layout.addWidget(self.file_group)

        params_group = QtWidgets.QGroupBox("转录参数（全参数）")
        params_layout = QtWidgets.QGridLayout(params_group)
        params_layout.setColumnStretch(1, 1)
        params_layout.setHorizontalSpacing(12)
        params_layout.setVerticalSpacing(10)

        self.api_key_edit = QtWidgets.QLineEdit()
        self.api_key_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.model_edit = QtWidgets.QLineEdit("openai/whisper-large-v3")
        self.language_edit = QtWidgets.QLineEdit("zh")
        self.prompt_edit = QtWidgets.QLineEdit()
        self.temperature_edit = QtWidgets.QLineEdit("0")

        self.response_format_combo = QtWidgets.QComboBox()
        self.response_format_combo.addItems(["text", "json", "srt", "vtt", "verbose_json"])
        self.response_format_combo.setCurrentText("verbose_json")

        self.timestamp_edit = QtWidgets.QLineEdit("segment")
        self.timeout_edit = QtWidgets.QLineEdit("300")
        self.retries_edit = QtWidgets.QLineEdit("2")
        self.max_chars_edit = QtWidgets.QLineEdit("20")
        self.max_duration_edit = QtWidgets.QLineEdit("8.0")
        self.max_silence_edit = QtWidgets.QLineEdit("1.0")

        self._add_grid_row(params_layout, 0, "API Key", self.api_key_edit)
        self._add_grid_row(params_layout, 1, "模型(model)", self.model_edit)
        self._add_grid_row(params_layout, 2, "语言(language)", self.language_edit)
        self._add_grid_row(params_layout, 3, "提示词(prompt)", self.prompt_edit)
        self._add_grid_row(params_layout, 4, "温度(temperature)", self.temperature_edit)
        self._add_grid_row(params_layout, 5, "返回格式(response_format)", self.response_format_combo)
        self._add_grid_row(
            params_layout,
            6,
            "时间粒度(timestamp_granularities)",
            self.timestamp_edit,
            hint="逗号分隔: segment,word",
        )
        self._add_grid_row(params_layout, 7, "请求超时(秒)", self.timeout_edit)
        self._add_grid_row(params_layout, 8, "重试次数", self.retries_edit)
        self._add_grid_row(params_layout, 9, "每段最大字符数", self.max_chars_edit)
        self._add_grid_row(params_layout, 10, "每段最长时间(秒)", self.max_duration_edit)
        self._add_grid_row(params_layout, 11, "每段最长停顿(秒)", self.max_silence_edit)

        main_layout.addWidget(params_group)

        output_group = QtWidgets.QGroupBox("输出")
        output_layout = QtWidgets.QHBoxLayout(output_group)
        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录（默认与音频同目录）")
        output_layout.addWidget(self.output_dir_edit)

        output_button = QtWidgets.QPushButton("选择输出目录")
        output_button.clicked.connect(self._choose_output)
        output_layout.addWidget(output_button)

        main_layout.addWidget(output_group)

        action_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("一键启动转录")
        self.start_button.clicked.connect(self._start)
        action_layout.addWidget(self.start_button)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        action_layout.addWidget(self.progress)
        main_layout.addLayout(action_layout)

        log_group = QtWidgets.QGroupBox("日志")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

    def _apply_theme(self) -> None:
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f6f7fb"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#eef2ff"))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#1f2937"))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#111827"))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                margin-top: 8px;
                padding: 12px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #111827;
            }
            QLabel { color: #1f2937; }
            QLineEdit, QComboBox, QTextEdit {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 6px 8px;
                background: #ffffff;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #2563eb;
            }
            QPushButton {
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:disabled {
                background: #93c5fd;
            }
            QProgressBar {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                text-align: center;
                background: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #60a5fa;
            }
            """
        )

    def _add_grid_row(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        label: str,
        widget: QtWidgets.QWidget,
        hint: Optional[str] = None,
    ) -> None:
        label_widget = QtWidgets.QLabel(label)
        layout.addWidget(label_widget, row, 0)
        layout.addWidget(widget, row, 1)
        if hint:
            hint_label = QtWidgets.QLabel(hint)
            hint_label.setStyleSheet("color: #6b7280;")
            layout.addWidget(hint_label, row, 2)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        local_path = urls[0].toLocalFile()
        if local_path and os.path.isfile(local_path):
            self.audio_path_edit.setText(local_path)
            self._log(f"已选择文件: {local_path}")

    def _choose_file(self) -> None:
        filetypes = "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg *.opus)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择音频文件", "", filetypes)
        if path:
            self.audio_path_edit.setText(path)
            self._log(f"已选择文件: {path}")

    def _choose_output(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_dir_edit.setText(path)
            self._log(f"输出目录: {path}")

    def _start(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            return

        audio_path = self.audio_path_edit.text().strip()
        if not audio_path:
            QtWidgets.QMessageBox.warning(self, "缺少文件", "请先选择音频文件。")
            return

        options = self._collect_options()
        if not options:
            return

        self.start_button.setEnabled(False)
        self.progress.setVisible(True)
        self._log("开始转录...")

        self.worker_thread = QtCore.QThread()
        self.worker = TranscriptionWorker(options)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._log)
        self.worker.finished.connect(self._finish_success)
        self.worker.failed.connect(self._finish_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _collect_options(self) -> Optional[TranscriptionOptions]:
        try:
            temperature = float(self.temperature_edit.text())
            timeout = int(self.timeout_edit.text())
            retries = int(self.retries_edit.text())
            max_chars = int(self.max_chars_edit.text())
            max_duration = float(self.max_duration_edit.text())
            max_silence = float(self.max_silence_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "参数错误", "请检查数值参数格式是否正确。")
            return None

        timestamps = [
            value.strip() for value in self.timestamp_edit.text().split(",") if value.strip()
        ]

        audio_path = self.audio_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or os.path.dirname(audio_path)

        return TranscriptionOptions(
            api_key=self.api_key_edit.text().strip(),
            model=self.model_edit.text().strip(),
            language=self.language_edit.text().strip(),
            prompt=self.prompt_edit.text().strip(),
            temperature=temperature,
            response_format=self.response_format_combo.currentText().strip(),
            timestamp_granularities=timestamps,
            timeout_seconds=timeout,
            max_retries=retries,
            max_chars_per_segment=max_chars,
            max_segment_duration=max_duration,
            max_silence_duration=max_silence,
            audio_path=audio_path,
            output_dir=output_dir,
        )

    def _finish_success(self, output_path: str) -> None:
        self._log(f"完成! 输出文件: {output_path}")
        QtWidgets.QMessageBox.information(self, "完成", f"转录完成!\n{output_path}")
        self._finish_cleanup()

    def _finish_error(self, message: str) -> None:
        self._log(f"错误: {message}")
        QtWidgets.QMessageBox.critical(self, "错误", message)
        self._finish_cleanup()

    def _finish_cleanup(self) -> None:
        self.progress.setVisible(False)
        self.start_button.setEnabled(True)

    def _cleanup_worker(self) -> None:
        if self.worker:
            self.worker.deleteLater()
        if self.worker_thread:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main() -> None:
    app = QtWidgets.QApplication([])
    window = TranscriptionWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
