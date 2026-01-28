import json
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import timedelta
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
                    raise RuntimeError(
                        f"API 返回错误 {response.status_code}: {response.text.strip()}"
                    )

                if options.response_format in {"srt", "vtt"}:
                    with open(output_path, "w", encoding="utf-8") as output_file:
                        output_file.write(response.text)
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

        lines = []
        for idx, seg in enumerate(merged, start=1):
            start_time = self._format_timestamp(seg.start)
            end_time = self._format_timestamp(seg.end)
            lines.append(str(idx))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(seg.text)
            lines.append("")

        return "\n".join(lines)

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
