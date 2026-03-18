from pathlib import Path
import importlib.util
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _missing_gui_dependencies() -> list[str]:
    module_checks = {
        "PySide6": "PySide6",
        "qdarktheme": "pyqtdarktheme",
    }
    missing = []
    for module_name, package_name in module_checks.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _pause_before_exit() -> None:
    if not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\n按回车键退出...")
    except EOFError:
        pass


def _print_dependency_help(missing: list[str]) -> None:
    packages = " ".join(missing)
    print(
        "缺少 GUI 运行依赖，请先安装：\n"
        f"  python -m pip install {packages}\n"
        "或直接执行 start.sh / start.bat 自动安装依赖。",
        file=sys.stderr,
    )


missing = _missing_gui_dependencies()
if missing:
    _print_dependency_help(missing)
    _pause_before_exit()
    raise SystemExit(1)

from deepinfra_transcriber.gui_app import main


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        print("启动失败，详细错误如下：", file=sys.stderr)
        traceback.print_exc()
        _pause_before_exit()
        raise SystemExit(1)
