"""对话级日志：记录每轮的步骤、工具调用、耗时与 token 用量（JSONL）。"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class DialogueLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _file(self) -> Path:
        return self.log_dir / f"{datetime.now():%Y-%m-%d}.jsonl"

    async def log(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        line = json.dumps(record, ensure_ascii=False, default=str)
        async with self._lock:
            await asyncio.to_thread(self._append, line)

    def _append(self, line: str) -> None:
        with self._file().open("a", encoding="utf-8") as f:
            f.write(line + "\n")
