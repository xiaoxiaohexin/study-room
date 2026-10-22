import pytest

from app.rag.chunker import Chunk, MarkdownChunker



def test_markdown_chunker_headers_and_overlap():
    """Existing test: verifies header hierarchy and basic splitting."""
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
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_chunk_model_expansion_defaults():
    """Verifies default values and explicit assignment in Chunk model."""
    c_default = Chunk(index=0, content="test", tokens=5)
    assert c_default.page_start == 1
    assert c_default.page_end == 1
    assert c_default.chapter is None
    assert c_default.headers == []

    c_custom = Chunk(
        index=1,
        content="test",
        tokens=5,
        headers=["Ch 1", "Sec 1"],
        page_start=42,
        page_end=43,
        chapter="Ch 1 > Sec 1",
    )
    assert c_custom.page_start == 42
    assert c_custom.page_end == 43
    assert c_custom.chapter == "Ch 1 > Sec 1"


def test_chunker_without_page_markers():
    """Verifies unpaged documents default gracefully to page 1."""
    chunker = MarkdownChunker()
    doc = """# 知识图谱简介
这是关于知识图谱的第一段介绍。

这是第二段介绍，深入解释实体与关系。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].chapter == "知识图谱简介"


def test_chunker_single_page_marker():
    """Verifies explicit single page marker tracking."""
    chunker = MarkdownChunker()
    doc = """<!-- PAGE: 5 -->
# 第五章 深度学习
深度学习的基础概念介绍段落。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].page_start == 5
    assert chunks[0].page_end == 5
    assert "<!-- PAGE" not in chunks[0].content


def test_chunker_multi_page_span_in_single_chunk():
    """Verifies chunk spanning across multiple pages calculates min and max."""
    chunker = MarkdownChunker(chunk_size_tokens=500)
    doc = """<!-- PAGE: 42 -->
这是第42页的内容段落。

<!-- PAGE: 43 -->
这是第43页的内容段落，紧跟在后。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].page_start == 42
    assert chunks[0].page_end == 43
    assert "<!-- PAGE" not in chunks[0].content


def test_chunker_page_transition_across_chunks():
    """Verifies page transitions correctly split into distinct chunks when token limit is reached."""
    chunker = MarkdownChunker(chunk_size_tokens=60, chunk_overlap_tokens=0)
    p1 = "第一页长段落 " * 6
    p2 = "第二页长段落 " * 6
    doc = f"""# 章节一
<!-- PAGE: 1 -->
{p1}

<!-- PAGE: 2 -->
{p2}
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 2
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 2


def test_chunker_sliding_window_overlap_preserves_page_range():
    """Verifies sliding window overlap retains page number of carried-over paragraphs."""
    chunker = MarkdownChunker(chunk_size_tokens=70, chunk_overlap_tokens=40)
    p1 = "段落一内容 " * 5
    p2 = "段落二内容 " * 5
    p3 = "段落三内容 " * 5
    doc = f"""<!-- PAGE: 1 -->
{p1}

<!-- PAGE: 2 -->
{p2}

<!-- PAGE: 3 -->
{p3}
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 2
    # Chunk 0 covers p1 (page 1) and p2 (page 2)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "段落一内容" in chunks[0].content
    assert "段落二内容" in chunks[0].content

    # Chunk 1 retains p2 (page 2) and adds p3 (page 3)
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 3
    assert "段落二内容" in chunks[1].content
    assert "段落三内容" in chunks[1].content


def test_chunker_form_feed_delimiter():
    """Verifies form feed (\\f) automatically increments page numbers."""
    chunker = MarkdownChunker()
    doc = "第一页文本段落。\n\f\n第二页文本段落。"
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "\f" not in chunks[0].content


def test_chunker_consecutive_empty_pages():
    """Verifies consecutive empty page markers do not emit empty chunks and update page index."""
    chunker = MarkdownChunker()
    doc = """<!-- PAGE: 1 -->
<!-- PAGE: 2 -->
<!-- PAGE: 3 -->
正文内容出现在第3页。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 3
    assert "正文内容出现在第3页。" in chunks[0].content


def test_chunker_chapter_hierarchy_formatting():
    """Verifies multi-level headers format chapter breadcrumb string correctly."""
    chunker = MarkdownChunker()
    doc = """# 第一篇 基础篇
## 第二章 网络协议
### 2.1 物理层
物理层传输介质与编码介绍。
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) == 1
    assert chunks[0].headers == ["第一篇 基础篇", "第二章 网络协议", "2.1 物理层"]
    assert chunks[0].chapter == "第一篇 基础篇 > 第二章 网络协议 > 2.1 物理层"


@pytest.mark.asyncio
async def test_document_loader_page_markers_and_fallbacks(tmp_path):
    """验证 DocumentLoader 正常注入物理页码与异常处理。"""
    from app.rag.loader import DocumentLoader

    loader = DocumentLoader()

    # 1. 纯文本/Markdown 默认第 1 页
    note_path = tmp_path / "test.md"
    note_path.write_text("# 绪论\n正文内容。", encoding="utf-8")
    res_md = await loader.load_to_markdown(note_path)
    assert "<!-- PAGE: 1 -->" in res_md
    assert "正文内容。" in res_md

    # 2. 分页符 \f 自动分页
    rfc_path = tmp_path / "rfc.txt"
    rfc_path.write_text("第一页段落\f第二页段落\f第三页段落", encoding="utf-8")
    res_rfc = await loader.load_to_markdown(rfc_path)
    assert "<!-- PAGE: 1 -->" in res_rfc
    assert "<!-- PAGE: 2 -->" in res_rfc
    assert "<!-- PAGE: 3 -->" in res_rfc

    # 3. 0 字节空文件保护
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    res_empty = await loader.load_to_markdown(empty_path)
    assert res_empty == ""

    # 4. 不存在文件保护
    res_nonexistent = await loader.load_to_markdown(tmp_path / "nonexistent.pdf")
    assert res_nonexistent == ""

