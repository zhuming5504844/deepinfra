import json
import math
import os
import textwrap
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import timedelta
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import requests
import pysrt

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    TkinterDnD = tk.Tk
    DND_FILES = None


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


class TranscriptionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DeepInfra 语音转录字幕")
        self.root.geometry("860x760")
        self.root.minsize(860, 760)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        file_frame = ttk.LabelFrame(main, text="音频文件", padding=12)
        file_frame.pack(fill=tk.X)

        self.audio_path_var = tk.StringVar()
        audio_entry = ttk.Entry(file_frame, textvariable=self.audio_path_var)
        audio_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        browse_btn = ttk.Button(file_frame, text="选择文件", command=self._choose_file)
        browse_btn.pack(side=tk.LEFT, padx=(8, 0))

        if DND_AVAILABLE:
            drop_hint = ttk.Label(file_frame, text="(可拖放文件)", foreground="#666")
            drop_hint.pack(side=tk.LEFT, padx=(8, 0))

        params_frame = ttk.LabelFrame(main, text="转录参数（全参数）", padding=12)
        params_frame.pack(fill=tk.X, pady=(12, 0))

        grid = ttk.Frame(params_frame)
        grid.pack(fill=tk.X)

        self.api_key_var = tk.StringVar()
        self.model_var = tk.StringVar(value="openai/whisper-large-v3")
        self.language_var = tk.StringVar(value="zh")
        self.prompt_var = tk.StringVar()
        self.temperature_var = tk.StringVar(value="0")
        self.response_format_var = tk.StringVar(value="verbose_json")
        self.timestamp_var = tk.StringVar(value="segment")
        self.timeout_var = tk.StringVar(value="300")
        self.retries_var = tk.StringVar(value="2")
        self.max_chars_var = tk.StringVar(value="20")
        self.max_duration_var = tk.StringVar(value="8.0")
        self.max_silence_var = tk.StringVar(value="1.0")

        self._add_row(grid, 0, "API Key", self.api_key_var, show="*")
        self._add_row(grid, 1, "模型(model)", self.model_var)
        self._add_row(grid, 2, "语言(language)", self.language_var)
        self._add_row(grid, 3, "提示词(prompt)", self.prompt_var)
        self._add_row(grid, 4, "温度(temperature)", self.temperature_var)

        response_choices = ["text", "json", "srt", "vtt", "verbose_json"]
        response_box = ttk.Combobox(
            grid, textvariable=self.response_format_var, values=response_choices, width=18
        )
        response_box.state(["readonly"])
        self._add_row(grid, 5, "返回格式(response_format)", widget=response_box)

        self._add_row(
            grid,
            6,
            "时间粒度(timestamp_granularities)",
            self.timestamp_var,
            hint="逗号分隔: segment,word",
        )
        self._add_row(grid, 7, "请求超时(秒)", self.timeout_var)
        self._add_row(grid, 8, "重试次数", self.retries_var)
        self._add_row(grid, 9, "每段最大字符数", self.max_chars_var)
        self._add_row(grid, 10, "每段最长时间(秒)", self.max_duration_var)
        self._add_row(grid, 11, "每段最长停顿(秒)", self.max_silence_var)

        output_frame = ttk.LabelFrame(main, text="输出", padding=12)
        output_frame.pack(fill=tk.X, pady=(12, 0))

        self.output_path_var = tk.StringVar()
        output_entry = ttk.Entry(output_frame, textvariable=self.output_path_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        output_btn = ttk.Button(output_frame, text="选择输出目录", command=self._choose_output)
        output_btn.pack(side=tk.LEFT, padx=(8, 0))

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(12, 0))

        self.start_button = ttk.Button(action_frame, text="一键启动转录", command=self._start)
        self.start_button.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(action_frame, mode="indeterminate")
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        log_frame = ttk.LabelFrame(main, text="日志", padding=12)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.log_text = tk.Text(log_frame, height=14)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._log("应用启动完成。")
        if not DND_AVAILABLE:
            self._log("提示: 未安装 tkinterdnd2，拖放功能不可用。")
        else:
            self._enable_drop()

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
            path = event.data.strip("{}")
            if os.path.isfile(path):
                self.audio_path_var.set(path)
                self._log(f"已选择文件: {path}")

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", handle_drop)

    def _choose_file(self) -> None:
        filetypes = [
            ("音频文件", "*.mp3 *.wav *.m4a *.flac *.ogg *.opus"),
            ("所有文件", "*.*"),
        ]
        path = filedialog.askopenfilename(title="选择音频文件", filetypes=filetypes)
        if path:
            self.audio_path_var.set(path)
            self._log(f"已选择文件: {path}")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_path_var.set(path)
            self._log(f"输出目录: {path}")

    def _start(self) -> None:
        if self.start_button["state"] == tk.DISABLED:
            return

        if not self.audio_path_var.get():
            messagebox.showwarning("缺少文件", "请先选择音频文件。")
            return

        options = self._collect_options()
        if not options:
            return

        self.start_button.config(state=tk.DISABLED)
        self.progress.start(10)
        self._log("开始转录...")

        thread = threading.Thread(target=self._run_transcription, args=(options,), daemon=True)
        thread.start()

    def _collect_options(self) -> Optional[TranscriptionOptions]:
        try:
            temperature = float(self.temperature_var.get())
            timeout = int(self.timeout_var.get())
            retries = int(self.retries_var.get())
            max_chars = int(self.max_chars_var.get())
            max_duration = float(self.max_duration_var.get())
            max_silence = float(self.max_silence_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "请检查数值参数格式是否正确。")
            return None

        timestamps = [
            value.strip() for value in self.timestamp_var.get().split(",") if value.strip()
        ]

        return TranscriptionOptions(
            api_key=self.api_key_var.get().strip(),
            model=self.model_var.get().strip(),
            language=self.language_var.get().strip(),
            prompt=self.prompt_var.get().strip(),
            temperature=temperature,
            response_format=self.response_format_var.get().strip(),
            timestamp_granularities=timestamps,
            timeout_seconds=timeout,
            max_retries=retries,
            max_chars_per_segment=max_chars,
            max_segment_duration=max_duration,
            max_silence_duration=max_silence,
        )

    def _run_transcription(self, options: TranscriptionOptions) -> None:
        try:
            output_path = self._transcribe(options)
            self._log(f"完成! 输出文件: {output_path}")
            messagebox.showinfo("完成", f"转录完成!\n{output_path}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"错误: {exc}")
            messagebox.showerror("错误", str(exc))
        finally:
            self.progress.stop()
            self.start_button.config(state=tk.NORMAL)

    def _transcribe(self, options: TranscriptionOptions) -> str:
        audio_path = self.audio_path_var.get()
        output_dir = self.output_path_var.get() or os.path.dirname(audio_path)
        os.makedirs(output_dir, exist_ok=True)

        headers = {"Authorization": f"Bearer {options.api_key}"} if options.api_key else {}

        model_name = self._normalize_model(options.model)
        if model_name != options.model:
            self._log(f"模型已自动补全为: {model_name}")

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

        filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_extension = "srt" if options.response_format in {"srt", "vtt"} else "txt"
        output_path = os.path.join(output_dir, f"{filename}.{output_extension}")

        for attempt in range(options.max_retries + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    files = {"file": (os.path.basename(audio_path), audio_file)}
                    self._log("正在请求 API...")
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
                    output_path = os.path.join(output_dir, f"{filename}.srt")
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
                self._log(f"请求失败，{wait_time} 秒后重试: {exc}")
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
        delta = timedelta(seconds=seconds)
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

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
