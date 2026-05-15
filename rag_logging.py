import importlib.util
import json
import sysconfig
import traceback
from datetime import datetime
from pathlib import Path


_STDLIB_LOGGING_PATH = Path(sysconfig.get_path("stdlib")) / "logging" / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("_stdlib_logging", _STDLIB_LOGGING_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load standard logging module from {_STDLIB_LOGGING_PATH}")

_STDLIB_LOGGING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STDLIB_LOGGING)

for _name in dir(_STDLIB_LOGGING):
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = getattr(_STDLIB_LOGGING, _name)


LOG_DIR = Path("storage")
LOG_FILE = LOG_DIR / "chat.log"


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def log_step(step_name: str, **values: object) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%H:%M:%S")
    details = " ".join(f"{key}={_format_value(value)}" for key, value in values.items())
    line = f"[{timestamp}] {step_name}"
    if details:
        line = f"{line} {details}"

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")


def log_error(step_name: str, error: Exception, **values: object) -> None:
    payload = dict(values)
    payload["error_type"] = type(error).__name__
    payload["error"] = str(error)
    payload["traceback"] = traceback.format_exc(limit=10)
    log_step(step_name, **payload)
