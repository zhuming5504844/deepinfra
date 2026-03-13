import json
import math
import os
import textwrap
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import requests

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    TkinterDnD = tk.Tk
    DND_FILES = None


API_URL = "https://api.deepinfra.com/v1/inference/openai/whisper-large-v3"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RETRIES = 2
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
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_RETRIES
    max_chars_per_segment: int = DEFAULT_MAX_CHARS_PER_SEGMENT
    max_segment_duration: float = DEFAULT_MAX_SEGMENT_DURATION
    max_silence_duration: float = DEFAULT_MAX_SILENCE_DURATION


class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DeepInfra 语音转录字幕")
        self.root.geometry("860x760")
        self.root.minsize(860, 760)

        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Action.TButton", padding=(10, 6))

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="音频文件", padding=12)
        file_frame.pack(fill=tk.X)

        self.audio_listbox = tk.Listbox(file_frame, height=5, selectmode=tk.EXTENDED)
        self.audio_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        audio_scroll = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.audio_listbox.yview)
        audio_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))
        self.audio_listbox.config(yscrollcommand=audio_scroll.set)

        file_btns = ttk.Frame(file_frame)
        file_btns.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(file_btns, text="添加文件", command=self._choose_files).pack(fill=tk.X)
        ttk.Button(file_btns, text="移除选中", command=self._remove_selected_files).pack(
            fill=tk.X, pady=(6, 0)
        )
        ttk.Button(file_btns, text="清空队列", command=self._clear_audio_files).pack(fill=tk.X, pady=(6, 0))

        if DND_AVAILABLE:
            self._enable_drop()

        params_frame = ttk.LabelFrame(main, text="转录参数（必填项）", padding=12)
        params_frame.pack(fill=tk.X, pady=(12, 0))

        grid = ttk.Frame(params_frame)
        grid.pack(fill=tk.X)

        self.api_key_var = tk.StringVar()
        self.api_url_var = tk.StringVar(value=API_URL)
        self.model_var = tk.StringVar(value="openai/whisper-large-v3")
        self.chunk_level_var = tk.StringVar(value="segment")

        self._add_row(grid, 0, "API Key", self.api_key_var, show="*")
        self._add_row(grid, 1, "API 地址", self.api_url_var)
        model_box = ttk.Combobox(
            grid,
            textvariable=self.model_var,
            values=["openai/whisper-large-v3", "openai/whisper-large-v2"],
        )
        self._add_row(grid, 2, "模型(model)", widget=model_box)

        chunk_box = ttk.Combobox(
            grid,
            textvariable=self.chunk_level_var,
            values=["segment", "word"],
            width=18,
        )
        chunk_box.state(["readonly"])
        self._add_row(grid, 3, "输出粒度", widget=chunk_box, hint="segment=段落, word=逐词")

        output_frame = ttk.LabelFrame(main, text="输出", padding=12)
        output_frame.pack(fill=tk.X, pady=(12, 0))
        output_frame.columnconfigure(0, weight=1)

        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        output_entry.grid(row=0, column=0, sticky=tk.EW)

        output_btn = ttk.Button(output_frame, text="选择输出目录", command=self._choose_output)
        output_btn.grid(row=0, column=1, padx=(8, 0))

        action_frame = ttk.LabelFrame(main, text="操作", padding=10)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        action_frame.columnconfigure(1, weight=1)

        button_group = ttk.Frame(action_frame)
        button_group.grid(row=0, column=0, sticky=tk.W)

        self.start_button = ttk.Button(
            button_group,
            text="一键启动转录",
            command=self._start,
            style="Action.TButton",
        )
        self.start_button.pack(side=tk.LEFT)

        save_btn = ttk.Button(
            button_group,
            text="保存默认设置",
            command=self._save_settings,
            style="Action.TButton",
        )
        save_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="准备就绪")
        status_label = ttk.Label(action_frame, textvariable=self.status_var, foreground="#555")
        status_label.grid(row=0, column=1, sticky=tk.E, padx=(12, 0))

        self.progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(8, 0))

        log_frame = ttk.LabelFrame(main, text="日志", padding=12)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.log_text = tk.Text(log_frame, height=14)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._log("应用启动完成。")
        if not DND_AVAILABLE:
            self._log("提示: 未安装 tkinterdnd2，拖放功能不可用。")

    def _add_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: Optional[tk.StringVar] = None,
        show: Optional[str] = None,
        hint: Optional[str] = None,
        widget: Optional[tk.Widget] = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        if widget is None:
            entry = ttk.Entry(parent, textvariable=variable, show=show)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=4)
            widget = entry
        else:
            widget.grid(row=row, column=1, sticky=tk.W, pady=4)

        if hint:
            ttk.Label(parent, text=hint, foreground="#666").grid(
                row=row, column=2, sticky=tk.W, padx=(8, 0), pady=4
            )

        parent.columnconfigure(1, weight=1)

    def _enable_drop(self) -> None:
        if not DND_AVAILABLE:
            return

        def handle_drop(event: tk.Event) -> None:
            dropped = self.root.tk.splitlist(event.data)
            self._add_audio_files(dropped)

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", handle_drop)

    def _choose_files(self) -> None:
        filetypes = [
            ("音频文件", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.wma *.amr *.webm *.mp4 *.mpeg *.mpga"),
            ("所有文件", "*.*"),
        ]
        paths = filedialog.askopenfilenames(title="选择音频文件", filetypes=filetypes)
        if paths:
            self._add_audio_files(paths)

    def _add_audio_files(self, paths) -> None:
        existing = set(self.audio_listbox.get(0, tk.END))
        added_count = 0
        for path in paths:
            normalized = path.strip("{}")
            if not os.path.isfile(normalized):
                continue
            if not self._is_supported_audio_file(normalized):
                self._log(f"已忽略不支持的格式: {normalized}")
                continue
            if normalized in existing:
                continue
            self.audio_listbox.insert(tk.END, normalized)
            existing.add(normalized)
            added_count += 1
            self._log(f"已添加文件: {normalized}")

    def _remove_selected_files(self) -> None:
        selected = list(self.audio_listbox.curselection())
        for index in reversed(selected):
            self.audio_listbox.delete(index)
        if selected:
            self._log(f"已移除 {len(selected)} 个文件。")

    def _clear_audio_files(self) -> None:
        count = self.audio_listbox.size()
        if count == 0:
            return
        self.audio_listbox.delete(0, tk.END)
        self._log("已清空文件队列。")

    def _get_audio_files(self) -> List[str]:
        return list(self.audio_listbox.get(0, tk.END))

    def _is_supported_audio_file(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in SUPPORTED_AUDIO_EXTENSIONS

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_path_var.set(path)
            self._log(f"输出目录: {path}")

    def _settings_snapshot(self) -> dict:
        return {
            "api_key": self.api_key_var.get().strip(),
            "api_url": self.api_url_var.get().strip(),
            "model": self.model_var.get().strip(),
            "chunk_level": self.chunk_level_var.get().strip(),
            "output_dir": self.output_path_var.get().strip(),
        }

    def _apply_settings(self, settings: dict) -> None:
        self.api_key_var.set(settings.get("api_key", self.api_key_var.get()))
        self.api_url_var.set(settings.get("api_url", self.api_url_var.get()))
        self.model_var.set(settings.get("model", self.model_var.get()))
        self.chunk_level_var.set(settings.get("chunk_level", self.chunk_level_var.get()))
        self.output_path_var.set(settings.get("output_dir", self.output_path_var.get()))

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
            messagebox.showerror("保存失败", f"无法保存默认设置: {exc}")

    def _start(self) -> None:
        if self.start_button["state"] == tk.DISABLED:
            return

        audio_files = self._get_audio_files()
        if not audio_files:
            messagebox.showwarning("缺少文件", "请先添加至少一个音频文件。")
            return

        options = self._collect_options()
        if not options:
            return

        self.start_button.config(state=tk.DISABLED)
        self.status_var.set("正在转录...")
        self.progress.start(10)
        self._log("开始转录...")

        thread = threading.Thread(
            target=self._run_transcription,
            args=(options, audio_files),
            daemon=True,
        )
        thread.start()

    def _collect_options(self) -> Optional[TranscriptionOptions]:
        api_key = self.api_key_var.get().strip()
        api_url = self.api_url_var.get().strip()
        model = self.model_var.get().strip()
        chunk_level = self.chunk_level_var.get().strip()

        if not api_key:
            messagebox.showwarning("缺少 API Key", "请输入 API Key。")
            return None
        if not api_url:
            messagebox.showwarning("缺少 API 地址", "请输入 API 地址。")
            return None
        if not model:
            messagebox.showwarning("缺少模型", "请输入或选择模型名称。")
            return None

        return TranscriptionOptions(
            api_key=api_key,
            api_url=api_url,
            model=model,
            chunk_level=chunk_level or "segment",
        )

    def _run_transcription(self, options: TranscriptionOptions, audio_files: List[str]) -> None:
        try:
            output_paths = []
            total = len(audio_files)
            for index, audio_path in enumerate(audio_files, start=1):
                self._log(f"[{index}/{total}] 开始处理: {audio_path}")
                output_path = self._transcribe(options, audio_path)
                output_paths.append(output_path)
                self._log(f"[{index}/{total}] 完成: {output_path}")
            summary = "\n".join(output_paths)
            messagebox.showinfo("完成", f"批量转录完成，共 {len(output_paths)} 个文件:\n{summary}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"错误: {exc}")
            messagebox.showerror("错误", str(exc))
        finally:
            self.progress.stop()
            self.start_button.config(state=tk.NORMAL)
            self.status_var.set("准备就绪")

    def _transcribe(self, options: TranscriptionOptions, audio_path: str) -> str:
        output_dir = self.output_path_var.get() or os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)

        headers = {"Authorization": f"Bearer {options.api_key}"}

        model_name = self._normalize_model(options.model)
        if model_name != options.model:
            self._log(f"模型已自动补全为: {model_name}")

        data = {
            "task": "transcribe",
            "chunk_level": options.chunk_level,
        }

        payload = {key: value for key, value in data.items() if value is not None}

        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{filename}.srt")

        api_url = self._build_api_url(options.api_url, model_name)

        for attempt in range(options.max_retries + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    files = {"audio": (os.path.basename(audio_path), audio_file)}
                    self._log("正在请求 API...")
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
                    self._resolve_async_response(api_url, headers, response.json(), options)
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
                self._log(f"请求失败，{wait_time} 秒后重试: {exc}")
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
        return self._build_srt_from_segments(
            result.get("segments", []),
            max_chars,
            max_duration,
            max_silence,
        )

    def _extract_transcription_result(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        output = payload.get("output")
        if isinstance(output, dict):
            return output
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                return first
        return payload

    def _resolve_async_response(
        self,
        api_url: str,
        headers: dict,
        payload: dict,
        options: TranscriptionOptions,
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
            self._log(f"任务状态: {status or 'unknown'}，{ASYNC_POLL_INTERVAL_SECONDS} 秒后轮询...")
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
        for idx, word in enumerate(words_data, start=1):
            text = (word.get("word") or "").strip()
            if not text:
                continue
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            srt_lines.extend(
                [
                    str(idx),
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

        srt_lines = []
        for idx, seg in enumerate(refined, start=1):
            srt_lines.extend(
                [
                    str(idx),
                    f"{self._format_timestamp(seg.start)} --> {self._format_timestamp(seg.end)}",
                    seg.text,
                    "",
                ]
            )
        return "\n".join(srt_lines).strip() + ("\n" if srt_lines else "")

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
            return (
                "API 返回错误 402: 账户余额不足，请在 DeepInfra 控制台充值或配置自动续费。"
            )
        if status == 401:
            return "API 返回错误 401: API Key 无效或缺失。"
        if status == 404:
            return (
                "API 返回错误 404: 模型不存在，请确认模型名称（如 openai/whisper-large-v3）。"
            )
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
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)


def main() -> None:
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
