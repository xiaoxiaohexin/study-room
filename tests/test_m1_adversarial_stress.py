"""Adversarial Empirical Stress Testing Suite for Milestone 1.

Targets:
- DocumentLoader (app/rag/loader.py)
- MarkdownChunker (app/rag/chunker.py)
- IngestionPipeline & RagRetriever (app/rag/pipeline.py, app/rag/retriever.py)
- REST endpoints (/api/rag/search & /api/v1/rag/search)

Verifies:
1. 1,000+ pages documents scaling & monotonicity
2. Missing page markers default & carryover behavior
3. Consecutive page markers with zero text (leading, trailing, interleaved, all-empty)
4. Non-standard page marker whitespace, casing, and colon variations
5. Massive single paragraphs exceeding token budgets (English, CJK, unpunctuated)
6. Multi-page spans crossing sliding window overlaps
7. Form feed (\f) edge cases & boundary transitions
8. DocumentLoader robustness (real 639-page PDF, synthetic 1,000-page PDF, encodings, blank files)
9. End-to-end vector pipeline & REST API page retrieval consistency
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
import pytest
from httpx import ASGITransport, AsyncClient
from pypdf import PdfWriter

from app.core.config import Settings
from app.core.container import AppContainer
from app.core.tokens import count_tokens
from app.llm.embedding import HashingEmbedder
from app.main import create_app
from app.rag.chunker import Chunk, MarkdownChunker
from app.rag.loader import DocumentLoader
from app.rag.pipeline import IngestionPipeline
from app.rag.retriever import RagRetriever
from app.storage.sqlite import SQLiteStore
from app.storage.vector import SQLiteVectorStore


# ============================================================================
# 1. 1,000+ Pages Scaling & Monotonicity
# ============================================================================

def test_stress_1000_pages_exact_page_monotonicity_and_no_exceptions():
    """Stress-test a 1,200-page document with sliding window overlap and headers."""
    chunker = MarkdownChunker(chunk_size_tokens=150, chunk_overlap_tokens=40)
    pages = []
    for p in range(1, 1201):
        pages.append(
            f"<!-- PAGE: {p} -->\n"
            f"# Chapter {p}\n"
            f"Physical page {p} content paragraph 1 discussing fundamental theorems.\n\n"
            f"Physical page {p} content paragraph 2 with detailed derivations."
        )
    doc = "\n\n".join(pages)

    t0 = time.perf_counter()
    chunks = chunker.split_text(doc)
    elapsed = time.perf_counter() - t0

    assert len(chunks) >= 1200, f"Expected at least 1200 chunks, got {len(chunks)}"
    assert elapsed < 5.0, f"Chunking 1200 pages should finish under 5s, took {elapsed:.2f}s"

    prev_start = 1
    for idx, c in enumerate(chunks):
        # 1. Page ranges must be valid
        assert 1 <= c.page_start <= c.page_end <= 1200, (
            f"Chunk {idx} invalid page bounds: {c.page_start}-{c.page_end}"
        )
        # 2. Monotonic start progression
        assert c.page_start >= prev_start, (
            f"Chunk {idx} page_start ({c.page_start}) regressed from previous ({prev_start})"
        )
        prev_start = c.page_start
        # 3. Clean comment stripping
        assert "<!-- PAGE" not in c.content, f"Unstripped marker in chunk {idx}"
        assert "<!--" not in c.content, f"HTML comment marker leak in chunk {idx}"
        assert "\f" not in c.content, f"Form feed leak in chunk {idx}"

    # Verify first and last chunk
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 1200


# ============================================================================
# 2. Missing Page Markers
# ============================================================================

def test_stress_missing_page_markers_defaults_and_section_carryover():
    """Verify completely unpaged documents default to 1, and partial markers carry over."""
    chunker = MarkdownChunker(chunk_size_tokens=100)

    # 2.1 Completely unpaged document with 30 sections
    unpaged_doc = "\n\n".join([
        f"# Section {i}\nParagraph {i} content text explaining core concepts."
        for i in range(1, 31)
    ])
    chunks = chunker.split_text(unpaged_doc)
    assert len(chunks) >= 10
    for idx, c in enumerate(chunks):
        assert c.page_start == 1 and c.page_end == 1, (
            f"Unpaged chunk {idx} expected page 1-1, got {c.page_start}-{c.page_end}"
        )

    # 2.2 Partial markers: Sec 1 unpaged (1), Sec 2 marked (77), Sec 3 unpaged (should remain 77)
    mixed_doc = """# Section 1
Content 1 without marker.

# Section 2
<!-- PAGE: 77 -->
Content 2 with marker 77.

# Section 3
Content 3 unpaged, carries over from section 2.
"""
    mixed_chunks = chunker.split_text(mixed_doc)
    assert len(mixed_chunks) == 3
    assert mixed_chunks[0].page_start == 1 and mixed_chunks[0].page_end == 1
    assert mixed_chunks[1].page_start == 77 and mixed_chunks[1].page_end == 77
    assert mixed_chunks[2].page_start == 77 and mixed_chunks[2].page_end == 77


# ============================================================================
# 3. Consecutive Empty Page Markers
# ============================================================================

def test_stress_consecutive_empty_page_markers_variations():
    """Test leading, trailing, middle consecutive empty page markers and all-empty docs."""
    chunker = MarkdownChunker(chunk_size_tokens=60, chunk_overlap_tokens=0)

    # 3.1 5 leading empty pages before text
    doc_leading = """<!-- PAGE: 1 -->
<!-- PAGE: 2 -->
<!-- PAGE: 3 -->
<!-- PAGE: 4 -->
<!-- PAGE: 5 -->
First text appears on page 5.
"""
    c_lead = chunker.split_text(doc_leading)
    assert len(c_lead) == 1
    assert c_lead[0].page_start == 5
    assert c_lead[0].page_end == 5
    assert "First text appears on page 5." in c_lead[0].content

    # 3.2 5 trailing empty pages after text
    doc_trailing = """<!-- PAGE: 10 -->
Text on page 10.
<!-- PAGE: 11 -->
<!-- PAGE: 12 -->
<!-- PAGE: 13 -->
<!-- PAGE: 14 -->
<!-- PAGE: 15 -->
"""
    c_trail = chunker.split_text(doc_trailing)
    assert len(c_trail) == 1
    assert c_trail[0].page_start == 10
    assert c_trail[0].page_end == 10
    assert "Text on page 10." in c_trail[0].content

    # 3.3 Middle empty pages with distinct chunks
    p1 = "Word " * 60
    p20 = "Item " * 60
    doc_middle = f"""<!-- PAGE: 1 -->
{p1}
<!-- PAGE: 2 -->
<!-- PAGE: 3 -->
<!-- PAGE: 4 -->
<!-- PAGE: 19 -->
<!-- PAGE: 20 -->
{p20}
"""
    c_mid = chunker.split_text(doc_middle)
    assert len(c_mid) == 2
    assert c_mid[0].page_start == 1 and c_mid[0].page_end == 1
    assert c_mid[1].page_start == 20 and c_mid[1].page_end == 20

    # 3.4 Entirely empty pages (50 page markers with zero text)
    doc_all_empty = "\n".join([f"<!-- PAGE: {i} -->" for i in range(1, 51)])
    c_empty = chunker.split_text(doc_all_empty)
    assert c_empty == [], "All-empty page marker document must yield zero chunks without exception"


# ============================================================================
# 4. Non-Standard Whitespace, Casing, and Colon Variations
# ============================================================================

def test_stress_non_standard_whitespace_casing_and_colons():
    """Verify regex tolerance for all casing, whitespace, and colon permutations."""
    chunker = MarkdownChunker()

    variations = [
        ("<!--PAGE:10-->\nNo whitespace", 10),
        ("<!--   PAGE:   20   -->\nMultiple spaces", 20),
        ("<!-- page: 30 -->\nLowercase", 30),
        ("<!-- PaGe: 40 -->\nMixed case", 40),
        ("<!-- PAGE:\t50\t-->\nTabs", 50),
        ("<!--\nPAGE:\n60\n-->\nNewlines inside marker", 60),
        ("<!-- PAGE 70 -->\nNo colon uppercase", 70),
        ("<!-- page 80 -->\nNo colon lowercase", 80),
        ("<!--   page   90   -->\nNo colon multi-space", 90),
        ("<!-- PAGE:000100 -->\nLeading zeros", 100),
    ]

    for raw_marker, expected_page in variations:
        chunks = chunker.split_text(raw_marker)
        assert len(chunks) == 1, f"Failed for {raw_marker}: expected 1 chunk"
        assert chunks[0].page_start == expected_page, (
            f"Failed for {raw_marker}: expected page {expected_page}, got {chunks[0].page_start}"
        )
        assert chunks[0].page_end == expected_page
        assert "<!--" not in chunks[0].content, f"Failed comment stripping on {raw_marker}"
        assert "PAGE" not in chunks[0].content

    # Inline without newlines
    inline_doc = "Paragraph one end.<!-- PAGE: 2 -->Paragraph two start."
    inline_chunks = chunker.split_text(inline_doc)
    assert len(inline_chunks) == 1
    assert inline_chunks[0].page_start == 1
    assert inline_chunks[0].page_end == 2
    assert "<!-- PAGE: 2 -->" not in inline_chunks[0].content


# ============================================================================
# 5. Massive Paragraphs & Token Budget Splitting
# ============================================================================

def test_stress_massive_paragraphs_and_boundary_splitting():
    """Test massive unpunctuated text, CJK text, and token limit handling."""
    chunker = MarkdownChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)

    # 5.1 Pure single massive English sentence without punctuation (2,000 words)
    giant_en = "Quantum " * 2000
    chunks_en = chunker.split_text(giant_en)
    assert len(chunks_en) >= 20
    for idx, c in enumerate(chunks_en):
        assert c.tokens <= 100, f"Chunk {idx} exceeded token limit: {c.tokens}"
        assert c.page_start == 1 and c.page_end == 1

    # 5.2 Pure single massive CJK sentence without punctuation (1,000 characters)
    giant_cjk = "量子纠缠是量子力学中一种奇特的物理现象两个粒子无论相隔多远都能瞬时相互影响" * 25
    chunks_cjk = chunker.split_text(giant_cjk)
    assert len(chunks_cjk) >= 10
    for idx, c in enumerate(chunks_cjk):
        assert c.tokens <= 100, f"CJK Chunk {idx} exceeded token limit: {c.tokens}"
        assert c.page_start == 1 and c.page_end == 1

    # 5.3 Massive punctuated English text (100 sentences of 30 words)
    punctuated_doc = " ".join([f"This is sentence number {i} with several words." for i in range(100)])
    chunks_punc = chunker.split_text(punctuated_doc)
    assert len(chunks_punc) >= 5
    for c in chunks_punc:
        assert c.tokens <= 120


# ============================================================================
# 6. Sliding Window Overlap Across Multi-Page Spans
# ============================================================================

def test_stress_sliding_window_overlap_multi_page_span():
    """Verify paragraphs retained in sliding window overlap accurately preserve their page numbers."""
    chunker = MarkdownChunker(chunk_size_tokens=150, chunk_overlap_tokens=60)

    # Page 1 has P1 (80 tokens)
    # Page 2 has P2 (50 tokens)
    # Page 3 has P3 (80 tokens)
    # Page 4 has P4 (50 tokens)
    p1 = "Alpha " * 80
    p2 = "Beta " * 50
    p3 = "Gamma " * 80
    p4 = "Delta " * 50

    doc = f"""<!-- PAGE: 1 -->
{p1}

<!-- PAGE: 2 -->
{p2}

<!-- PAGE: 3 -->
{p3}

<!-- PAGE: 4 -->
{p4}
"""
    chunks = chunker.split_text(doc)
    assert len(chunks) >= 2

    # Chunk 0: P1 (p1) + P2 (p2) => page_start=1, page_end=2
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "Alpha" in chunks[0].content
    assert "Beta" in chunks[0].content

    # Chunk 1: retains P2 (p2) through overlap, adds P3 (p3) => page_start=2, page_end=3
    assert chunks[1].page_start == 2
    assert chunks[1].page_end == 3
    assert "Beta" in chunks[1].content
    assert "Gamma" in chunks[1].content


# ============================================================================
# 7. Form Feed Edge Cases
# ============================================================================

def test_stress_form_feed_edge_cases():
    """Test multiple consecutive \\f, leading/trailing \\f, and mixed with <!-- PAGE: N -->."""
    chunker = MarkdownChunker()

    # 7.1 Multiple consecutive form feeds
    doc_ff = "First page text.\f\f\fFourth page text."
    chunks_ff = chunker.split_text(doc_ff)
    assert len(chunks_ff) == 1
    assert chunks_ff[0].page_start == 1
    assert chunks_ff[0].page_end == 4
    assert "\f" not in chunks_ff[0].content

    # 7.2 Leading form feeds
    doc_lead_ff = "\f\fThird page text."
    chunks_lead = chunker.split_text(doc_lead_ff)
    assert len(chunks_lead) == 1
    assert chunks_lead[0].page_start == 3
    assert chunks_lead[0].page_end == 3
    assert "\f" not in chunks_lead[0].content

    # 7.3 Interleaved form feed and page marker
    doc_interleaved = "Page 1.\n\f\nPage 2.\n<!-- PAGE: 50 -->\nPage 50."
    chunks_int = chunker.split_text(doc_interleaved)
    assert len(chunks_int) == 1
    assert chunks_int[0].page_start == 1
    assert chunks_int[0].page_end == 50
    assert "<!-- PAGE" not in chunks_int[0].content
    assert "\f" not in chunks_int[0].content


# ============================================================================
# 8. DocumentLoader Robustness & File Encoding
# ============================================================================

@pytest.mark.asyncio
async def test_stress_document_loader_1000_pages_pdf_and_encodings(tmp_path: Path):
    """Test synthetic 1,000-page PDF, encodings, and edge file conditions in DocumentLoader."""
    loader = DocumentLoader()
    tmp_path.mkdir(parents=True, exist_ok=True)

    # 8.1 Synthetic 1,000-page blank PDF
    writer = PdfWriter()
    for _ in range(1000):
        writer.add_blank_page(width=100, height=100)
    pdf_path = tmp_path / "synthetic_1000p.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    md_pdf = await loader.load_to_markdown(pdf_path)
    assert isinstance(md_pdf, str)
    assert md_pdf.count("<!-- PAGE:") == 1000
    assert "<!-- PAGE: 1000 -->" in md_pdf

    # 8.2 Real repository PDF (Hello-Agents 639 pages)
    real_pdf = Path("data/uploads/Hello-Agents-V1.0.0-20251103-水印.pdf")
    if real_pdf.exists():
        md_real = await loader.load_to_markdown(real_pdf)
        assert len(md_real) > 100000
        assert "<!-- PAGE: 1 -->" in md_real
        assert "<!-- PAGE: 639 -->" in md_real

        # Chunk the real document
        chunker = MarkdownChunker(chunk_size_tokens=500, chunk_overlap_tokens=80)
        chunks = chunker.split_text(md_real)
        assert len(chunks) > 500
        for c in chunks:
            assert 1 <= c.page_start <= c.page_end <= 639
            assert "<!-- PAGE" not in c.content

    # 8.3 Multiple encodings: GBK and Latin-1
    tmp_path.mkdir(parents=True, exist_ok=True)
    gbk_file = tmp_path / "test_gbk.txt"
    gbk_file.write_bytes("这是一个GBK编码的计算机体系结构教材。\n第二段内容。".encode("gbk"))
    md_gbk = await loader.load_to_markdown(gbk_file)
    assert "计算机体系结构" in md_gbk
    assert "<!-- PAGE: 1 -->" in md_gbk

    tmp_path.mkdir(parents=True, exist_ok=True)
    latin_file = tmp_path / "test_latin.txt"
    latin_file.write_bytes("Théorie des graphes et applications.".encode("latin-1"))
    md_latin = await loader.load_to_markdown(latin_file)
    assert "des graphes" in md_latin
    assert "<!-- PAGE: 1 -->" in md_latin


# ============================================================================
# 9. Full RAG Pipeline & REST API Retrieval Consistency
# ============================================================================

@pytest.mark.asyncio
async def test_stress_full_rag_pipeline_and_rest_search_endpoints(tmp_path: Path):
    """Verify ingestion of a multi-page document into vector store and REST search via both routes."""
    settings = Settings(
        offline_mode=True,
        sqlite_path=tmp_path / "stress_rag.db",
        data_dir=tmp_path / "stress_data",
    )
    app = create_app(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container

    try:
        # Create a 50-page document
        doc_path = tmp_path / "distributed_systems.md"
        doc_sections = []
        for p in range(1, 51):
            doc_sections.append(
                f"<!-- PAGE: {p} -->\n"
                f"# 第{p}章 分布式一致性协议\n"
                f"这是第{p}页的内容，讨论 Paxos 与 Raft 算法的Leader选举机制与状态机复制。"
            )
        doc_path.write_text("\n\n".join(doc_sections), encoding="utf-8")

        # Ingest through pipeline
        doc_id, chunk_count = await container.ingestion_pipeline.ingest_file(doc_path, namespace="stress_ns")
        assert chunk_count >= 50
        assert doc_id is not None

        # Test both REST endpoints
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for route in ("/api/rag/search", "/api/v1/rag/search"):
                resp = await ac.post(route, json={"query": "Raft 算法 Leader 选举", "namespace": "stress_ns", "top_k": 3})
                assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"
                data = resp.json()
                assert data["count"] >= 1
                for item in data["results"]:
                    assert "page_start" in item
                    assert "page_end" in item
                    assert 1 <= item["page_start"] <= item["page_end"] <= 50
                    assert item["filename"] == "distributed_systems.md"

        # Verify DocChunk citation generation
        chunks = await container.rag_retriever.search(["Paxos 算法"], namespace="stress_ns", top_k=2)
        assert len(chunks) >= 1
        top_chunk = chunks[0]
        assert top_chunk.page_label.startswith("P.")
        citation = top_chunk.format_citation()
        assert "distributed_systems.md" in citation
        assert top_chunk.page_label in citation
        prompt_block = container.rag_retriever.format_prompt_citations(chunks)
        assert "[DOCUMENT:" in prompt_block
    finally:
        await container.aclose()
