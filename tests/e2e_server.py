"""Launch the backend for full-stack Playwright E2E tests.

Runs in offline mode (FakeLLM + in-memory vector store) so no llama-server or
Qdrant is required. Data is isolated under ``data_e2e/`` and wiped on start.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data_e2e"
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR, ignore_errors=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["OFFLINE_MODE"] = "true"
os.environ["SQLITE_PATH"] = str(DATA_DIR / "agent_e2e.db")
os.environ["DATA_DIR"] = str(DATA_DIR)
os.environ["LOG_LEVEL"] = "WARNING"
# 联网导入 E2E：允许回环（本地 fixture 网页服务器），运行期自动联网补充关闭
os.environ["WEB_ENABLED"] = "true"
os.environ["WEB_ALLOW_LOOPBACK"] = "true"
os.environ["WEB_FALLBACK_MODE"] = "off"
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("E2E_BACKEND_PORT", "8010"))
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
