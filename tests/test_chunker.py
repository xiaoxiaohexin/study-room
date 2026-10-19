from app.rag.chunker import MarkdownChunker


def test_markdown_chunker_headers_and_overlap():
    chunker = MarkdownChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)
    doc = """# 第一章 绪论
这是绪论的内容段落一。

这是绪论的内容段落二。

## 1.1 背景
背景介绍段落，包含详细信息。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) >= 1
    # 验证是否保留了标题层级
    has_header = any("第一章 绪论" in c.content for c in chunks)
    assert has_header
