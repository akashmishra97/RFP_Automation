from pathlib import Path
import json
import asyncio
from typing import Any, Dict, List, Optional


def ensure_dirs(base_out: Path) -> Path:
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "workflow_output").mkdir(parents=True, exist_ok=True)
    return base_out / "workflow_output"


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def sleep_backoff(attempt: int, base_delay: int = 5) -> asyncio.Future:
    return asyncio.sleep(base_delay * max(1, attempt))
