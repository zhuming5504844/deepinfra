import json
import math
import os
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import qdarktheme
import requests
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

API_URL = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RETRIES = 2
DEFAULT_BATCH_PARALLEL_JOBS = 3
ASYNC_POLL_INTERVAL_SECONDS = 2
ASYNC_POLL_TIMEOUT_SECONDS = 300
DEFAULT_MAX_CHARS_PER_SEGMENT = 20
DEFAULT_MAX_SEGMENT_DURATION = 8.0
DEFAULT_MAX_SILENCE_DURATION = 1.0
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".deepinfra_transcriber.json")
SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".amr",
    ".webm",
    ".mp4",
    ".mpeg",
    ".mpga",
}


def _apply_dark_theme(app: QApplication) -> None:
    if hasattr(qdarktheme, "setup_theme"):
        qdarktheme.setup_theme("auto")
    else:
        load_stylesheet = getattr(qdarktheme, "load_stylesheet", None)
        if callable(load_stylesheet):
            try:
                stylesheet = load_stylesheet()
            except TypeError:
                stylesheet = load_stylesheet("dark")
            if stylesheet:
                app.setStyleSheet(stylesheet)

    base_stylesheet = """
    QWidget { font-size: 13px; }
    QMainWindow { background: transparent; }
    QFrame#pageContainer {
        border-radius: 22px;
        background-color: rgba(255, 255, 255, 0.02);
    }
    QFrame#heroCard {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        background-color: rgba(120, 120, 120, 0.08);
    }
    QFrame#sectionCard {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        background-color: rgba(255, 255, 255, 0.03);
    }
    QFrame#statCard {
        border-radius: 14px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    QLabel#heroTitle { font-size: 24px; font-weight: 700; }
    QLabel#heroSubtitle { color: palette(mid); font-size: 13px; }
    QLabel#sectionTitle { font-size: 15px; font-weight: 700; }
    QLabel#sectionDescription { color: palette(mid); }
    QLabel#statValue { font-size: 18px; font-weight: 700; }
    QLabel#statLabel { color: palette(mid); }
    QPushButton {
        min-height: 36px;
        padding: 0 14px;
        border-radius: 10px;
    }
    QPushButton#primaryButton { font-weight: 700; }
    QLineEdit, QComboBox, QListWidget, QPlainTextEdit {
        border-radius: 12px;
        padding: 6px 8px;
    }
    QListWidget, QPlainTextEdit {
        background-color: rgba(0, 0, 0, 0.12);
    }
    QProgressBar {
        min-height: 12px;
        border-radius: 6px;
        text-align: center;
    }
    QProgressBar::chunk { border-radius: 6px; }
    """
    app.setStyleSheet(f"{app.styleSheet()}\n{base_stylesheet}")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionOptions:
    api_key: str
    api_url: str
    model: str
    chunk_level: str
    output_dir: str
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_RETRIES
    max_chars_per_segment: int = DEFAULT_MAX_CHARS_PER_SEGMENT
    max_segment_duration: float = DEFAULT_MAX_SEGMENT_DURATION
    max_silence_duration: float = DEFAULT_MAX_SILENCE_DURATION


class DropListWidget(QListWidget):
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class WorkerSignals(QObject):
    log = Signal(str)
    status = Signal(str)
    finished = Signal(list)
    error = Signal(str)


class BatchTranscriptionWorker(QRunnable):
    def __init__(self, app: "TranscriptionApp", options: TranscriptionOptions, audio_files: List[str]) -> None:
        super().__init__()
        self.app = app
        self.options = options
        self.audio_files = audio_files
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            output_paths = [""] * len(self.audio_files)
            total = len(self.audio_files)
            self.signals.log.emit(f"已启用批量模式，并行任务数: {DEFAULT_BATCH_PARALLEL_JOBS}")
            with ThreadPoolExecutor(max_workers=DEFAULT_BATCH_PARALLEL_JOBS) as executor:
                future_map = {}
                for index, audio_path in enumerate(self.audio_files, start=1):
                    self.signals.log.emit(f"[{index}/{total}] 已加入批处理队列: {audio_path}")
                    future = executor.submit(self.app._transcribe, self.options, audio_path, self.signals)
                    future_map[future] = index - 1

                completed = 0
                for future in as_completed(future_map):
                    output_path = future.result()
                    item_index = future_map[future]
                    output_paths[item_index] = output_path
                    completed += 1
                    self.signals.status.emit(f"正在转录... {completed}/{total}")
                    self.signals.log.emit(f"[{completed}/{total}] 完成: {output_path}")

            self.signals.finished.emit(output_paths)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class TranscriptionApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepInfra 语音转录字幕")
        self.resize(1380, 860)
        self.setMinimumSize(1200, 760)
        self.thread_pool = QThreadPool.globalInstance()
        self._worker: Optional[BatchTranscriptionWorker] = None
        self._build_ui()
        self._load_settings()
        self._refresh_overview()
        self._log("应用启动完成。")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(0)

        page = QFrame()
        page.setObjectName("pageContainer")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(16)
        root_layout.addWidget(page)

        page_layout.addWidget(self._build_header_card())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 5)
        page_layout.addWidget(splitter, stretch=1)

    def _build_header_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("heroCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("DeepInfra Whisper 批量转录工作台")
        title.setObjectName("heroTitle")
        subtitle = QLabel("新增现代化双栏布局、顶部标题区、原生日志面板与参数分组区域，便于后续继续扩展更多转录能力。")
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)
        self.queue_stat = self._create_stat_card("0", "待处理文件")
        self.output_stat = self._create_stat_card("SRT", "默认输出格式")
        self.parallel_stat = self._create_stat_card(str(DEFAULT_BATCH_PARALLEL_JOBS), "并行任务数")
        stats_layout.addWidget(self.queue_stat)
        stats_layout.addWidget(self.output_stat)
        stats_layout.addWidget(self.parallel_stat)
        layout.addLayout(stats_layout)
        return card

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_file_section(), stretch=4)
        layout.addWidget(self._build_form_section(), stretch=5)
        layout.addWidget(self._build_action_section(), stretch=2)
        return container

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(self._build_log_section(), stretch=1)
        return container

    def _build_file_section(self) -> QWidget:
        card, body = self._make_group(
            "音频队列",
            "支持拖拽导入音频文件，适合批量转录与字幕导出。",
            QHBoxLayout,
        )
        self.audio_list = DropListWidget()
        self.audio_list.setMinimumHeight(220)
        self.audio_list.files_dropped.connect(self._add_audio_files)
        body.addWidget(self.audio_list, stretch=1)

        actions = QVBoxLayout()
        actions.setSpacing(10)
        self.add_file_button = QPushButton("添加文件")
        self.add_file_button.clicked.connect(self._choose_files)
        actions.addWidget(self.add_file_button)

        self.remove_button = QPushButton("移除选中")
        self.remove_button.clicked.connect(self._remove_selected_files)
        actions.addWidget(self.remove_button)

        self.clear_button = QPushButton("清空队列")
        self.clear_button.clicked.connect(self._clear_audio_files)
        actions.addWidget(self.clear_button)

        tip = QLabel("拖拽文件到左侧列表即可加入任务队列。")
        tip.setObjectName("sectionDescription")
        tip.setWordWrap(True)
        actions.addWidget(tip)
        actions.addStretch(1)
        body.addLayout(actions)
        return card

    def _build_form_section(self) -> QWidget:
        card, body = self._make_group(
            "参数配置",
            "将常用参数拆分为多个分组区域，方便扩展更多模型与高级选项。",
            QVBoxLayout,
        )

        required_frame = QFrame()
        required_layout = QFormLayout(required_frame)
        required_layout.setContentsMargins(0, 0, 0, 0)
        required_layout.setHorizontalSpacing(14)
        required_layout.setVerticalSpacing(12)
        required_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("输入 DeepInfra API Key")
        required_layout.addRow("API Key", self.api_key_input)

        self.api_url_input = QLineEdit(API_URL)
        self.api_url_input.setPlaceholderText("推理接口地址")
        required_layout.addRow("API 地址", self.api_url_input)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["openai/whisper-large-v3", "openai/whisper-large-v2"])
        required_layout.addRow("模型", self.model_combo)
        body.addWidget(self._wrap_subsection("核心参数", "转录请求必须填写的基础信息。", required_frame))

        output_grid = QGridLayout()
        output_grid.setContentsMargins(0, 0, 0, 0)
        output_grid.setHorizontalSpacing(12)
        output_grid.setVerticalSpacing(12)

        self.chunk_level_combo = QComboBox()
        self.chunk_level_combo.addItems(["segment", "word"])
        output_grid.addWidget(QLabel("输出粒度"), 0, 0)
        output_grid.addWidget(self.chunk_level_combo, 0, 1)

        chunk_hint = QLabel("segment = 段落字幕，word = 逐词对齐。")
        chunk_hint.setObjectName("sectionDescription")
        output_grid.addWidget(chunk_hint, 1, 0, 1, 2)

        self.output_path_input = QLineEdit()
        self.output_path_input.setPlaceholderText("为空时默认输出到音频所在目录")
        output_grid.addWidget(QLabel("输出目录"), 2, 0)
        output_grid.addWidget(self.output_path_input, 2, 1)

        self.output_button = QPushButton("选择输出目录")
        self.output_button.clicked.connect(self._choose_output)
        output_grid.addWidget(self.output_button, 3, 1, alignment=Qt.AlignRight)
        body.addWidget(self._wrap_subsection("导出与字幕", "控制转录粒度与字幕保存位置。", output_grid))

        return card

    def _build_action_section(self) -> QWidget:
        card, body = self._make_group(
            "执行控制",
            "保留原生 Qt 进度条与状态提示，便于后续扩展取消任务、历史记录等功能。",
            QVBoxLayout,
        )

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        self.start_button = QPushButton("一键启动转录")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setMinimumHeight(42)
        self.start_button.clicked.connect(self._start)
        buttons.addWidget(self.start_button, stretch=2)

        self.save_button = QPushButton("保存默认设置")
        self.save_button.setMinimumHeight(42)
        self.save_button.clicked.connect(self._save_settings)
        buttons.addWidget(self.save_button, stretch=1)
        body.addLayout(buttons)

        status_row = QHBoxLayout()
        status_caption = QLabel("当前状态")
        status_caption.setObjectName("sectionDescription")
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_row.addWidget(status_caption)
        status_row.addWidget(self.status_label)
        body.addLayout(status_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        body.addWidget(self.progress)
        return card

    def _build_log_section(self) -> QWidget:
        card, body = self._make_group(
            "运行日志",
            "使用原生日志面板实时展示导入、请求、重试与完成状态，便于排查问题。",
            QVBoxLayout,
        )
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.log_text.setFont(font)
        self.log_text.setPlaceholderText("这里会显示详细的运行日志…")
        body.addWidget(self.log_text, stretch=1)
        return card

    def _create_stat_card(self, value: str, label: str) -> QFrame:
        card = QFrame()
        card.setObjectName("statCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(4)
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        text_label = QLabel(label)
        text_label.setObjectName("statLabel")
        layout.addWidget(value_label)
        layout.addWidget(text_label)
        card.value_label = value_label
        return card

    def _wrap_subsection(self, title: str, description: str, content) -> QFrame:
        frame = QFrame()
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        desc_label = QLabel(description)
        desc_label.setObjectName("sectionDescription")
        desc_label.setWordWrap(True)
        inner.addWidget(title_label)
        inner.addWidget(desc_label)

        if isinstance(content, QWidget):
            inner.addWidget(content)
        else:
            inner.addLayout(content)
        return frame

    def _make_group(self, title: str, description: str, body_layout_cls=QVBoxLayout):
        group = QFrame()
        group.setObjectName("sectionCard")
        wrapper = QVBoxLayout(group)
        wrapper.setContentsMargins(18, 18, 18, 18)
        wrapper.setSpacing(14)

        header = QLabel(title)
        header.setObjectName("sectionTitle")
        desc = QLabel(description)
        desc.setObjectName("sectionDescription")
        desc.setWordWrap(True)
        wrapper.addWidget(header)
        wrapper.addWidget(desc)

        body_layout = body_layout_cls()
        wrapper.addLayout(body_layout)
        return group, body_layout

    def _refresh_overview(self) -> None:
        self.queue_stat.value_label.setText(str(self.audio_list.count()))
        self.output_stat.value_label.setText("SRT")
        self.parallel_stat.value_label.setText(str(DEFAULT_BATCH_PARALLEL_JOBS))

    def _choose_files(self) -> None:
        file_filter = (
            "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.wma *.amr *.webm *.mp4 *.mpeg *.mpga);;所有文件 (*.*)"
        )
        paths, _ = QFileDialog.getOpenFileNames(self, "选择音频文件", "", file_filter)
        if paths:
            self._add_audio_files(paths)

    def _add_audio_files(self, paths: List[str]) -> None:
        existing = set(self._get_audio_files())
        added_count = 0
        for path in paths:
            normalized = path.strip().strip("{}")
            if not normalized or not os.path.isfile(normalized):
                continue
            if not self._is_supported_audio_file(normalized):
                self._log(f"已忽略不支持的格式: {normalized}")
                continue
            if normalized in existing:
                continue
            self.audio_list.addItem(QListWidgetItem(normalized))
            existing.add(normalized)
            added_count += 1
            self._log(f"已添加文件: {normalized}")
        if added_count:
            self._refresh_overview()

    def _remove_selected_files(self) -> None:
        selected = self.audio_list.selectedItems()
        for item in selected:
            row = self.audio_list.row(item)
            self.audio_list.takeItem(row)
        if selected:
            self._log(f"已移除 {len(selected)} 个文件。")
            self._refresh_overview()

    def _clear_audio_files(self) -> None:
        count = self.audio_list.count()
        if count == 0:
            return
        self.audio_list.clear()
        self._refresh_overview()
        self._log("已清空文件队列。")

    def _get_audio_files(self) -> List[str]:
        return [self.audio_list.item(index).text() for index in range(self.audio_list.count())]

    def _is_supported_audio_file(self, path: str) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_path_input.setText(path)
            self._log(f"输出目录: {path}")

    def _settings_snapshot(self) -> dict:
        return {
            "api_key": self.api_key_input.text().strip(),
            "api_url": self.api_url_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "chunk_level": self.chunk_level_combo.currentText().strip(),
            "output_dir": self.output_path_input.text().strip(),
        }

    def _apply_settings(self, settings: dict) -> None:
        self.api_key_input.setText(settings.get("api_key", self.api_key_input.text()))
        self.api_url_input.setText(settings.get("api_url", self.api_url_input.text()))
        self.model_combo.setCurrentText(settings.get("model", self.model_combo.currentText()))
        self.chunk_level_combo.setCurrentText(settings.get("chunk_level", self.chunk_level_combo.currentText()))
        self.output_path_input.setText(settings.get("output_dir", self.output_path_input.text()))

    def _load_settings(self) -> None:
        if not os.path.isfile(SETTINGS_PATH):
            return
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as settings_file:
                settings = json.load(settings_file)
            if isinstance(settings, dict):
                self._apply_settings(settings)
                self._log("已加载默认设置。")
        except Exception as exc:  # noqa: BLE001
            self._log(f"默认设置加载失败: {exc}")

    def _save_settings(self) -> None:
        settings = self._settings_snapshot()
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as settings_file:
                json.dump(settings, settings_file, ensure_ascii=False, indent=2)
            self._log("已保存默认设置。")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", f"无法保存默认设置: {exc}")

    def _set_running_state(self, is_running: bool) -> None:
        self.start_button.setEnabled(not is_running)
        self.add_file_button.setEnabled(not is_running)
        self.remove_button.setEnabled(not is_running)
        self.clear_button.setEnabled(not is_running)
        self.output_button.setEnabled(not is_running)
        if is_running:
            self.progress.setRange(0, 0)
            self.status_label.setText("正在转录...")
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.status_label.setText("准备就绪")

    def _start(self) -> None:
        audio_files = self._get_audio_files()
        if not audio_files:
            QMessageBox.warning(self, "缺少文件", "请先添加至少一个音频文件。")
            return

        options = self._collect_options()
        if not options:
            return

        self._set_running_state(True)
        self._log("开始转录...")
        worker = BatchTranscriptionWorker(self, options, audio_files)
        worker.signals.log.connect(self._log)
        worker.signals.status.connect(self.status_label.setText)
        worker.signals.error.connect(self._handle_worker_error)
        worker.signals.finished.connect(self._handle_worker_finished)
        self.thread_pool.start(worker)
        self._worker = worker

    def _collect_options(self) -> Optional[TranscriptionOptions]:
        api_key = self.api_key_input.text().strip()
        api_url = self.api_url_input.text().strip()
        model = self.model_combo.currentText().strip()
        chunk_level = self.chunk_level_combo.currentText().strip() or "segment"
        output_dir = self.output_path_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "缺少 API Key", "请输入 API Key。")
            return None
        if not api_url:
            QMessageBox.warning(self, "缺少 API 地址", "请输入 API 地址。")
            return None
        if not model:
            QMessageBox.warning(self, "缺少模型", "请输入或选择模型名称。")
            return None

        return TranscriptionOptions(
            api_key=api_key,
            api_url=api_url,
            model=model,
            chunk_level=chunk_level,
            output_dir=output_dir,
        )

    def _handle_worker_finished(self, output_paths: List[str]) -> None:
        self._set_running_state(False)
        summary = "\n".join(output_paths)
        QMessageBox.information(self, "完成", f"批量转录完成，共 {len(output_paths)} 个文件:\n{summary}")

    def _handle_worker_error(self, message: str) -> None:
        self._log(f"错误: {message}")
        self._set_running_state(False)
        QMessageBox.critical(self, "错误", message)

    def _transcribe(self, options: TranscriptionOptions, audio_path: str, signals: WorkerSignals) -> str:
        output_dir = options.output_dir or os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)

        headers = {"Authorization": f"Bearer {options.api_key}"}

        model_name = self._normalize_model(options.model)
        if model_name != options.model:
            signals.log.emit(f"模型已自动补全为: {model_name}")

        payload = {"task": "transcribe", "chunk_level": options.chunk_level}
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.srt")
        api_url = self._build_api_url(options.api_url, model_name)

        for attempt in range(options.max_retries + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    files = {"audio": (os.path.basename(audio_path), audio_file)}
                    signals.log.emit(f"正在请求 API: {audio_path}")
                    response = requests.post(
                        api_url,
                        headers=headers,
                        data=payload,
                        files=files,
                        timeout=options.timeout_seconds,
                    )
                if response.status_code >= 400:
                    raise RuntimeError(self._format_api_error(response))

                result = self._extract_transcription_result(
                    self._resolve_async_response(api_url, headers, response.json(), options, signals)
                )
                srt = self._build_srt(
                    result,
                    options.chunk_level,
                    options.max_chars_per_segment,
                    options.max_segment_duration,
                    options.max_silence_duration,
                )
                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(srt)
                return output_path
            except Exception as exc:  # noqa: BLE001
                if attempt >= options.max_retries:
                    raise exc
                wait_time = 2 ** attempt
                signals.log.emit(f"请求失败，{wait_time} 秒后重试: {exc}")
                time.sleep(wait_time)

        raise RuntimeError("无法完成转录，请检查网络或 API Key。")

    def _build_srt(
        self,
        result: dict,
        chunk_level: str,
        max_chars: int,
        max_duration: float,
        max_silence: float,
    ) -> str:
        if chunk_level == "word":
            words = result.get("words", [])
            if words:
                return self._build_srt_from_words(words)
        return self._build_srt_from_segments(result.get("segments", []), max_chars, max_duration, max_silence)

    def _extract_transcription_result(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        output = payload.get("output")
        if isinstance(output, dict):
            return output
        if isinstance(output, list) and output and isinstance(output[0], dict):
            return output[0]
        return payload

    def _resolve_async_response(
        self,
        api_url: str,
        headers: dict,
        payload: dict,
        options: TranscriptionOptions,
        signals: WorkerSignals,
    ) -> dict:
        if not isinstance(payload, dict):
            return payload

        status = str(payload.get("status", "")).lower()
        if status in {"", "completed", "succeeded", "success", "done"}:
            return payload

        status_url = payload.get("status_url") or payload.get("poll_url")
        job_id = payload.get("id") or payload.get("request_id")
        if not status_url and job_id:
            status_url = f"{api_url.rstrip('/')}/{job_id}"
        if not status_url:
            return payload

        deadline = time.time() + ASYNC_POLL_TIMEOUT_SECONDS
        while time.time() < deadline:
            signals.log.emit(f"任务状态: {status or 'unknown'}，{ASYNC_POLL_INTERVAL_SECONDS} 秒后轮询...")
            time.sleep(ASYNC_POLL_INTERVAL_SECONDS)
            poll_resp = requests.get(status_url, headers=headers, timeout=options.timeout_seconds)
            if poll_resp.status_code >= 400:
                raise RuntimeError(self._format_api_error(poll_resp))
            payload = poll_resp.json()
            if not isinstance(payload, dict):
                return payload
            status = str(payload.get("status", "")).lower()
            if status in {"completed", "succeeded", "success", "done"}:
                return payload
            if status in {"failed", "error", "cancelled"}:
                raise RuntimeError(f"异步任务失败，状态: {status}")

        raise TimeoutError(f"异步任务轮询超时（>{ASYNC_POLL_TIMEOUT_SECONDS} 秒）")

    def _build_srt_from_words(self, words_data: List[dict]) -> str:
        srt_lines = []
        line_number = 0
        for word in words_data:
            text = (word.get("word") or "").strip()
            if not text:
                continue
            line_number += 1
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            srt_lines.extend(
                [
                    str(line_number),
                    f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}",
                    text,
                    "",
                ]
            )
        return "\n".join(srt_lines).strip() + ("\n" if srt_lines else "")

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
            segments.append(Segment(start=float(segment.get("start", 0.0)), end=float(segment.get("end", 0.0)), text=text))

        merged: List[Segment] = []
        current: Optional[Segment] = None
        for seg in segments:
            if current is None:
                current = Segment(seg.start, seg.end, seg.text)
                continue

            gap = seg.start - current.end
            combined_text = f"{current.text} {seg.text}".strip()
            duration = seg.end - current.start
            if len(combined_text) <= max_chars and duration <= max_duration and gap <= max_silence:
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

        srt_lines = []
        for idx, seg in enumerate(refined, start=1):
            srt_lines.extend([str(idx), f"{self._format_timestamp(seg.start)} --> {self._format_timestamp(seg.end)}", seg.text, ""])
        return "\n".join(srt_lines).strip() + ("\n" if srt_lines else "")

    def _split_segment(self, segment: Segment, max_chars: int, max_duration: float) -> List[Segment]:
        text = segment.text.strip()
        if not text:
            return []

        if max_chars <= 0:
            chunks = [text]
        else:
            chunks = textwrap.wrap(text, width=max_chars, break_long_words=True, break_on_hyphens=False)

        total_duration = max(segment.end - segment.start, 0.0)
        if max_duration > 0 and total_duration > 0:
            required_count = max(len(chunks), math.ceil(total_duration / max_duration))
            if required_count > len(chunks):
                target_width = max(1, math.ceil(len(text) / required_count))
                if max_chars > 0:
                    target_width = min(max_chars, target_width)
                chunks = textwrap.wrap(text, width=target_width, break_long_words=True, break_on_hyphens=False)
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
        normalized = [chunk.strip() for chunk in chunks if chunk.strip()] or [""]
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
            replacements = [part.strip() for part in (first, second) if part.strip()]
            if not replacements:
                break
            normalized[longest_index : longest_index + 1] = replacements
        return [chunk for chunk in normalized if chunk]

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

    def _format_timestamp(self, seconds: float) -> str:
        total_ms = max(int(seconds * 1000), 0)
        hours = total_ms // 3_600_000
        minutes = (total_ms % 3_600_000) // 60_000
        secs = (total_ms % 60_000) // 1_000
        milliseconds = total_ms % 1_000
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

    def _build_api_url(self, api_url: str, model: str) -> str:
        base = api_url.strip().rstrip("/")
        if not base:
            return API_URL
        normalized_model = self._normalize_model(model)
        if base.endswith(normalized_model):
            return base
        if base.endswith("/v1/inference"):
            return f"{base}/{normalized_model}"
        if "/v1/inference/" in base:
            prefix = base.split("/v1/inference/")[0]
            return f"{prefix}/v1/inference/{normalized_model}"
        return base

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")


def main() -> int:
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)
    assert app is not None
    app.setApplicationName("deepinfra-transcriber")
    _apply_dark_theme(app)
    window = TranscriptionApp()
    window.show()
    result = app.exec()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
