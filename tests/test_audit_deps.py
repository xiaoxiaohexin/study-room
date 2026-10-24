"""审计 P0-2 回归：PDF 解析依赖必须声明，未使用的依赖必须移除。"""

from __future__ import annotations

from pathlib import Path


def test_pyproject_declares_pdf_dependencies():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "pypdf" in text
    assert "pypdfium2" in text
    assert "rank-bm25" not in text
    assert "jieba" not in text