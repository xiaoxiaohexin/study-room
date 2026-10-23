"""多格式文档统一加载转换引擎：支持 PDF 物理页码提取、多编码文本、图片多模态 OCR 与 MarkItDown 兜底。"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import pypdf
from pypdf.errors import PyPdfError

from app.llm.base import LLMClient
from app.observability.logging import get_logger
from app.schemas.messages import Message

try:
    import pypdfium2 as _pdfium
except ImportError:  # pragma: no cover - optional rendering backend
    _pdfium = None

log = get_logger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
_TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv", ".py", ".sh", ".yaml", ".yml", ".sql"}
_PAGE_MARKER_RE = re.compile(r"<!--\s*PAGE:\s*\d+\s*-->", re.IGNORECASE)

# 逐页判定：字符数低于该阈值的页面视为“无文本层/扫描页”，按预算交给多模态模型 OCR。
_MIN_PAGE_TEXT_CHARS = 10
# 单个文档每次入库最多 OCR 的页数预算，避免混合扫描件把整本书都送进视觉模型。
_OCR_MAX_PAGES_PER_DOC = 40
_OCR_CONCURRENCY = 4
_OCR_RENDER_SCALE = 2.0
_OCR_MAX_TOKENS = 4000
_OCR_PAGE_TIMEOUT = 600.0
_OCR_PROMPT = (
    "请将这一页教科书完整转写为 Markdown：保留数学公式的 LaTeX、标题层级、正文、"
    "例题与习题编号。不要输出图片链接或占位符，不要添加任何解释。"
)


class DocumentLoader:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm
        self._has_markitdown = False
        try:
            import markitdown  # noqa: F401

            self._has_markitdown = True
            log.info("markitdown_available")
        except ImportError:
            log.info("markitdown_not_available_using_builtins")

    async def load_to_markdown(self, file_path: Path) -> str:
        """将文档（PDF/Word/Excel/图片/TXT/MD/代码）转换为带有物理页码标记的 Markdown 字符串。"""
        if not file_path.exists():
            log.error("document_file_not_found", path=str(file_path))
            return ""

        try:
            if file_path.stat().st_size == 0:
                log.warning("empty_document_file", path=str(file_path))
                return ""
        except OSError as e:
            log.error("document_stat_failed", path=str(file_path), error=str(e))
            return ""

        log.info("loading_document", path=str(file_path))
        suffix = file_path.suffix.lower()

        # 1. PDF 文档：优先使用 pypdf 进行物理页码保留提取
        if suffix == ".pdf":
            return await self._load_pdf(file_path)

        # 2. 图片文件：通过多模态 LLM 提取视觉与 OCR 文字
        if suffix in _IMAGE_SUFFIXES and self.llm:
            try:
                resp = await self.llm.chat(
                    [
                        Message(
                            role="user",
                            content="请将这张图片的内容完整转换为结构化 Markdown，包含所有可见文字（OCR）、图表结构与视觉细节描述。",
                            images=[str(file_path)],
                        )
                    ]
                )
                desc = (resp.content or "").strip()
                return f"# 图片文档：{file_path.name}\n\n<!-- PAGE: 1 -->\n{desc}\n"
            except Exception as e:  # noqa: BLE001
                log.warning("llm_image_convert_failed", path=str(file_path), error=str(e))

        # 3. 纯文本 / 代码 / Markdown 文件
        if suffix in _TEXT_SUFFIXES:
            return await asyncio.to_thread(self._load_text, file_path)

        # 4. 二进制文档（Word/Excel/PPT 等）尝试 markitdown 解析
        if self._has_markitdown:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                res = await asyncio.to_thread(md.convert, str(file_path))
                text = (res.text_content or "").strip()
                log.info("markitdown_converted_successfully", path=str(file_path), char_count=len(text))
                if text:
                    if not _PAGE_MARKER_RE.search(text):
                        return f"<!-- PAGE: 1 -->\n\n{text}"
                    return text
            except Exception as e:  # noqa: BLE001
                log.warning("markitdown_convert_failed", path=str(file_path), error=str(e))

        # 5. 兜底纯文本读取
        return await asyncio.to_thread(self._load_text, file_path)

    async def _load_pdf(self, file_path: Path) -> str:
        """逐页提取文字；无文本层的扫描页交给多模态模型 OCR，并保留物理页码标记。"""
        try:
            page_texts = await asyncio.to_thread(self._extract_pdf_texts, file_path)
        except (PyPdfError, Exception) as e:  # noqa: BLE001
            log.warning("pypdf_load_failed", path=str(file_path), error=str(e))
            return await asyncio.to_thread(self._pdf_fallback, file_path)

        total_pages = len(page_texts)
        if total_pages == 0:
            log.warning("pdf_has_zero_pages", path=str(file_path))
            return ""

        # 逐页检测：只要个别页面没有文本层（教材里穿插的扫描插图页/习题页），
        # 就单独 OCR 这些页，而不是等整本书都缺文本才触发。
        missing = [
            i for i, t in enumerate(page_texts, start=1) if len(t.strip()) < _MIN_PAGE_TEXT_CHARS
        ]
        text_pages = total_pages - len(missing)

        if missing:
            ocr_pages = missing[:_OCR_MAX_PAGES_PER_DOC]
            if len(missing) > len(ocr_pages):
                log.warning(
                    "pdf_ocr_budget_truncated",
                    path=str(file_path),
                    pages_needing_ocr=len(missing),
                    pages_ocr_this_run=len(ocr_pages),
                )
            log.warning(
                "pdf_pages_without_text_detected",
                path=str(file_path),
                total_pages=total_pages,
                text_pages=text_pages,
                pages_needing_ocr=len(missing),
            )
            ocr_text = await self._ocr_pdf_pages(file_path, ocr_pages)
            for page_no, text in ocr_text.items():
                page_texts[page_no - 1] = text
            text_pages = sum(1 for t in page_texts if len(t.strip()) >= _MIN_PAGE_TEXT_CHARS)
            if text_pages == 0:
                log.warning("pdf_ocr_produced_no_text", path=str(file_path))
                return await asyncio.to_thread(self._pdf_fallback, file_path)

        page_blocks = [
            f"<!-- PAGE: {i} -->\n{text}"
            for i, text in enumerate(page_texts, start=1)
            if text.strip()
        ]
        if not page_blocks:
            return await asyncio.to_thread(self._pdf_fallback, file_path)

        result_md = "\n\n".join(page_blocks)
        log.info(
            "pdf_loaded_successfully",
            path=str(file_path),
            total_pages=total_pages,
            extracted_pages=len(page_blocks),
            char_count=len(result_md),
        )
        return result_md

    def _extract_pdf_texts(self, file_path: Path) -> list[str]:
        """返回与物理页一一对应的提取文本列表（无文本层的页为空串）。"""
        reader = pypdf.PdfReader(str(file_path))
        if reader.is_encrypted:
            try:
                # 尝试用空密码解密（常见于只设限制打印但可阅读的文档）
                reader.decrypt("")
            except Exception as dec_err:  # noqa: BLE001
                log.warning("pdf_decrypt_failed", path=str(file_path), error=str(dec_err))

        texts: list[str] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                texts.append((page.extract_text() or "").strip())
            except Exception as page_err:  # noqa: BLE001
                log.warning(
                    "pdf_page_extract_failed", path=str(file_path), page=page_idx, error=str(page_err)
                )
                texts.append("")
        return texts

    async def _ocr_pdf_pages(self, file_path: Path, page_numbers: list[int]) -> dict[int, str]:
        """渲染指定物理页并调用多模态模型转写为 Markdown，返回 {页码: 文本}。"""
        if self.llm is None or _pdfium is None or not page_numbers:
            log.warning(
                "pdf_ocr_unavailable",
                path=str(file_path),
                has_llm=self.llm is not None,
                has_renderer=_pdfium is not None,
            )
            return {}

        total = len(page_numbers)
        log.info(
            "pdf_ocr_started",
            path=str(file_path),
            pages_total=total,
            concurrency=_OCR_CONCURRENCY,
        )
        results: dict[int, str] = {}
        done = 0
        semaphore = asyncio.Semaphore(_OCR_CONCURRENCY)

        with tempfile.TemporaryDirectory(prefix="rag_ocr_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            async def _ocr_one(page_no: int) -> tuple[int, str]:
                async with semaphore:
                    try:
                        image = await asyncio.to_thread(
                            self._render_pdf_page, file_path, page_no, tmp_dir
                        )
                    except Exception as render_err:  # noqa: BLE001
                        log.warning("pdf_page_render_failed", page=page_no, error=str(render_err))
                        return page_no, ""
                    try:
                        resp = await self.llm.chat(
                            [Message(role="user", content=_OCR_PROMPT, images=[str(image)])],
                            max_tokens=_OCR_MAX_TOKENS,
                            timeout=_OCR_PAGE_TIMEOUT,
                        )
                        return page_no, (resp.content or "").strip()
                    except Exception as ocr_err:  # noqa: BLE001
                        log.warning("pdf_page_ocr_failed", page=page_no, error=str(ocr_err))
                        return page_no, ""
                    finally:
                        image.unlink(missing_ok=True)

            tasks = [
                asyncio.create_task(_ocr_one(page_no), name=f"ocr_page_{page_no}")
                for page_no in page_numbers
            ]
            for finished in asyncio.as_completed(tasks):
                page_no, text = await finished
                results[page_no] = text
                done += 1
                if done % 10 == 0 or done == total:
                    log.info("pdf_ocr_progress", done=done, total=total)

        transcribed = sum(1 for t in results.values() if t)
        log.info("pdf_ocr_completed", pages_total=total, pages_transcribed=transcribed)
        return {page_no: text for page_no, text in results.items() if text}

    def _render_pdf_page(self, file_path: Path, page_no: int, out_dir: Path) -> Path:
        """将单个物理页渲染为 PNG，供多模态模型识别。"""
        out_path = out_dir / f"page_{page_no:05d}.png"
        doc = _pdfium.PdfDocument(str(file_path))
        try:
            doc[page_no - 1].render(scale=_OCR_RENDER_SCALE).to_pil().save(out_path, format="PNG")
        finally:
            doc.close()
        return out_path

    def _pdf_fallback(self, file_path: Path) -> str:
        """文本层缺失时的兜底：尝试 markitdown，最后退化为纯页码标记。"""
        if self._has_markitdown:
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                res = md.convert(str(file_path))
                text = (res.text_content or "").strip()
                if text:
                    log.info("markitdown_pdf_fallback_success", path=str(file_path), length=len(text))
                    return f"<!-- PAGE: 1 -->\n\n{text}"
            except Exception as fb_err:  # noqa: BLE001
                log.warning("markitdown_pdf_fallback_failed", path=str(file_path), error=str(fb_err))
        try:
            total_pages = max(1, len(pypdf.PdfReader(str(file_path)).pages))
        except Exception:  # noqa: BLE001
            return self._load_text(file_path)
        return "\n\n".join(f"<!-- PAGE: {i} -->" for i in range(1, total_pages + 1))

    def _load_text(self, file_path: Path) -> str:
        """读取纯文本或 Markdown，自适应多种编码，并确保页码规范。"""
        content = ""
        for enc in ("utf-8", "gb18030", "gbk", "latin-1"):
            try:
                with open(file_path, encoding=enc) as f:
                    content = f.read()
                    break
            except (UnicodeDecodeError, OSError):
                continue

        if not content:
            try:
                with open(file_path, errors="ignore") as f:
                    content = f.read()
            except OSError as e:
                log.warning("read_text_fallback_failed", path=str(file_path), error=str(e))
                return ""

        content = content.strip()
        if not content:
            return ""

        # 若包含标准分页符 \f，按分页符拆分为多个页面
        if "\f" in content:
            parts = content.split("\f")
            page_blocks: list[str] = []
            for idx, part in enumerate(parts, start=1):
                part_text = part.strip()
                if part_text:
                    page_blocks.append(f"<!-- PAGE: {idx} -->\n{part_text}")
            return "\n\n".join(page_blocks)

        # 若已包含 <!-- PAGE: N --> 标记，保持原样
        if _PAGE_MARKER_RE.search(content):
            return content

        # 默认作为单页文档
        return f"<!-- PAGE: 1 -->\n\n{content}"

    # ==================== 物理页原文读取（供前端溯源预览） ====================

    def get_page_count(self, file_path: Path) -> int:
        """返回文档物理页数（PDF 取真实页数，其他按分页标记推断，至少 1 页）。"""
        if not file_path.exists():
            return 0
        if file_path.suffix.lower() == ".pdf":
            try:
                return max(1, len(pypdf.PdfReader(str(file_path)).pages))
            except Exception as e:  # noqa: BLE001
                log.warning("pdf_page_count_failed", path=str(file_path), error=str(e))
                return 1
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            raw = ""
        return max(1, len(_PAGE_MARKER_RE.findall(raw)))

    def read_page_text_sync(self, file_path: Path, page: int) -> tuple[str, int]:
        """同步读取第 ``page`` 页的纯文本，返回 (text, total_pages)。"""
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        page = max(1, int(page))
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            reader = pypdf.PdfReader(str(file_path))
            total = max(1, len(reader.pages))
            if page > total:
                raise ValueError(f"page {page} out of range (total {total})")
            try:
                text = (reader.pages[page - 1].extract_text() or "").strip()
            except Exception as e:  # noqa: BLE001
                log.warning("pdf_page_extract_failed", path=str(file_path), page=page, error=str(e))
                text = ""
            return text, total

        # Text / markdown documents: split by physical page markers when present.
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            raw = ""
        matches = list(_PAGE_MARKER_RE.finditer(raw))
        if not matches:
            return raw.strip(), 1
        preamble = raw[: matches[0].start()].strip()
        if preamble:
            # Content before the first marker belongs to page 1; markers then address
            # pages 2..N (common when authors only mark later pages).
            total = len(matches) + 1
            if page == 1:
                return preamble, total
            index = page - 2
        else:
            total = len(matches)
            index = page - 1
        if index < 0 or index >= len(matches):
            raise ValueError(f"page {page} out of range (total {total})")
        start = matches[index].end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        return raw[start:end].strip(), total

