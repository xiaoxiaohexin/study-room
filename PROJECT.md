# Project: General-Purpose Agent Framework & Infinite Knowledge Canvas

## Core Closed-Loop & Data Deletion Architecture
- **Closed-Loop Control System**: 一个**“基于 RAG 检索知识源生成测试 -> 自动化混合裁决评估 -> 基于记忆衰减（SM-2）计算下一次重放时间 -> 自动收集失败任务并编排优化流水线”**的闭环控制系统。
- **Data Deletion & State Mutation Policy**: **删数据的逻辑：Agent 端严禁硬删除，业务端采用软状态标记（Soft Resolve），仅在底层维护受用户作用域隔离的级联物理删除。**

## Architecture
- **Backend RAG & Document Processing**: `app/rag/loader.py`, `chunker.py`, `pipeline.py`, `retriever.py`. Ingests source documents (PDF/Markdown), preserves exact physical page numbers (`page_start`/`page_end`) and section hierarchy, vectorizes and stores with page-aware metadata.
- **Backend Storage, Knowledge Graph & Closed-Loop Engine**: `app/storage/sqlite.py`, `app/learning/store.py`, `app/learning/graph.py`, `app/learning/algorithms.py`, `app/tools/builtin/learning_tools.py`. Manages SQLite schemas: `knowledge_points` (provenance, raw excerpt, versions), `kp_relations` (typed semantic edges: `prerequisite`, `causal`, `similar`, `branch_followup`), `kp_prereq`, `mastery` (SM-2 interval/EF + EMA node stability), `mistakes` (soft-resolved failure regression queue), and `messages` (`parent_id` message tree).
- **Backend REST API**: `app/api/routes/rag.py`, `learn.py`, `llama_compat.py`. Exposes `/api/v1/rag/search` alias, `/api/courses/{course_id}/knowledge` endpoints (CRUD, revisions, rollback, relations sync), and normalized tool execution.
- **Frontend Chat & Selection**: `llama-webui/tools/ui/src/lib/components/app/chat/ChatMessages/`. Listens to text selection on assistant responses, displays floating toolbar with "提取为知识节点" / "追加到已有节点".
- **Frontend Canvas Workspace**: `InfiniteCanvasWorkspace.svelte`. Renders dual-zone cards (`kind: 'knowledge'`, immutable quote excerpt zone + editable user annotation zone), provenance badges with two-way click navigation (chat message scroll/highlight + PDF page preview), typed semantic edges (`CanvasEdge`), and version history drawer.
- **Frontend Message Tree & Branching**: `database.service.ts`, `branching.ts`, `chatStore`, `conversationsStore`. Implements true sub-branch follow-up from canvas nodes using `pendingBranchParentId`, branching tree visualization, and lossless branch switching.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | RAG Physical Page Extraction | `DocumentLoader` extracts PDF pages using `pypdf` with `<!-- PAGE: N -->` delimiters | M1 | ORIGINAL_REQUEST §R2 |
| 2 | Page-Aware Chunking | `MarkdownChunker` detects page markers, records `page_start` & `page_end` on chunks | M1 | ORIGINAL_REQUEST §R2 |
| 3 | Page-Aware Vector Payload & Citation | `IngestionPipeline` & `RagRetriever` return `page_start`/`page_end` in search and prompt citations (`P.42`) | M1 | ORIGINAL_REQUEST §R2 |
| 4 | RAG API Routing Alias | Support `/api/v1/rag/search` routing alias matching frontend expectations | M1 | Survey Backend |
| 5 | Knowledge Node Schema Migration | Idempotently migrate `knowledge_points` table with `raw_quote`, `content`, `version`, `provenance_json`, `history_json` | M2 | ORIGINAL_REQUEST §R1, R5 |
| 6 | Semantic Relations Schema (`kp_relations`) | SQLite table `kp_relations` supporting `prerequisite`, `causal`, `similar`, `branch_followup` | M2 | ORIGINAL_REQUEST §R3 |
| 7 | Dual-Write Prerequisite Sync | Dual-write `prerequisite` relations to `kp_prereq` and `GraphStore` for backward compatibility | M2 | ORIGINAL_REQUEST §R3 |
| 8 | Stability Enrollment & Knowledge APIs | `upsert_knowledge_point` auto-enrolls into `mastery`, handles provenance, revisions, and rollback REST endpoints | M2 | ORIGINAL_REQUEST §R1, R5 |
| 9 | Text Selection Floating Toolbar | Text selection in chat assistant messages triggers floating action menu to extract/append knowledge cards | M3 | ORIGINAL_REQUEST §R1 |
| 10 | Dual-Zone Knowledge Canvas Card | Card with immutable read-only excerpt zone and editable user content zone, displaying provenance badges | M3 | ORIGINAL_REQUEST §R1, R2, R5 |
| 11 | Bi-directional Chat & Document Anchors | Clicking chat badge scrolls & highlights message; clicking document badge previews PDF page | M3 | ORIGINAL_REQUEST §R2 |
| 12 | Strong Semantic Canvas Edges | Upgraded `CanvasEdge` with distinct SVG styles and menu for `prerequisite`, `causal`, `similar`, `branch_followup` | M4 | ORIGINAL_REQUEST §R3 |
| 13 | Edge Persistence & SQLite Sync | Syncing canvas semantic edges with backend `kp_relations` and `kp_prereq` | M4 | ORIGINAL_REQUEST §R3 |
| 14 | Node Branch Follow-up Engine | Branching follow-up from canvas node setting `pendingBranchParentId` to origin message ID in Dexie tree | M5 | ORIGINAL_REQUEST §R4 |
| 15 | Canvas Branch Indicator & Lossless Switching | Canvas turn cards show a branch indicator/switcher and switch branches via `conversationsStore.navigateToSibling` | M5 + Fix P1 | ORIGINAL_REQUEST §R4 |
| 16 | Revision History & Non-Destructive Rollback | Card revision badge `v{N} · 历史`, history timeline modal, and one-click rollback | M5 | ORIGINAL_REQUEST §R5 |
| 17 | E2E Test Suite (Tiers 1-4) | Comprehensive automated test suite covering all features across Tier 1 (Feature), Tier 2 (Boundary), Tier 3 (Cross-feature), Tier 4 (Workload) | E2E-Track / M6 | ORIGINAL_REQUEST §Verification |
| 18 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing and edge case hardening | M6 | Project Pattern |
| 19 | Scoped Data Ownership Isolation | `knowledge_points.user_id` + domain auto-registration, user/domain scoped CRUD, cross-domain write/lookup rejected, relation endpoint consistency checks | Fix P0-A | Review findings |
| 20 | Tool-Layer Scope Isolation | `ToolContext.rag_namespace/course_id` propagated from session; RAG/quiz/knowledge tools cannot select another namespace or user scope | Fix P0-B | Review findings |
| 21 | Real Provenance & Page Preview | RAG document registry, `/api/rag/documents/{id}/file|page`, quote attribution (`/api/rag/attribute`), real PDF page rendering via pdf.js | Fix P0-C/D | Review findings |
| 22 | Frontend Scope Wiring | `learning-context` single source for domain/namespace/user, real turn indices, canvas as single knowledge writer, backend hydration, backend-first versions | Fix P0-D/P1 | Review findings |
| 23 | Optional API Token Boundary | `API_TOKEN` opt-in bearer auth for all endpoints except `/health`; WebUI reuses API Key header | Fix P2 | Review findings |
| 24 | Full-Stack Browser E2E | Offline FakeLLM backend + real Vite dev server Playwright suite (selection→card→anchor, branch switching, real PDF page preview, web import→distill→promote) | Fix P2/P3 | Review findings |
| 25 | Web Import (A channel) | URL/search discovery → SSRF-checked fetch → readability extraction → staged snapshot in temporary KB | FIX-P3 | Staging request |
| 26 | Temporary Knowledge Base (Staging) | `documents.tier/source_type/url/content_hash`, isolated `staging_{embedder}` collection, per-session dedupe/capacity, 7-day TTL cleanup, never part of standard RAG | FIX-P3 | Staging request |
| 27 | Distillation to Knowledge Candidates | Quality score + LLM structured extraction (title/summary/content/raw_quote) with rule fallback → pending candidate inbox | FIX-P3 | Staging request |
| 28 | Promotion & Revoke | Promote staged snapshot + candidates into target domain namespace and knowledge nodes (canvas hydrates them); one-click revoke removes standard doc + created KPs | FIX-P3 | Staging request |
| 29 | Runtime Web Supplement (B channel) | RAG absolute-similarity gate → MCP search / provider → fetch + stage + `[WEB]` prompt block with URL citations (auto mode, off by default) | FIX-P3 | Staging request |
| 30 | WebTool over MCP | `web_tool` in ToolRegistry therefore exposed via `/tools` and `/mcp` (search/fetch/stage) with session scope | FIX-P3 | Staging request |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite Track | Complete opaque-box automated test suite (Tiers 1-4, 177 tests), TEST_INFRA.md & TEST_READY.md | none | DONE |
| M1 | Backend RAG Physical Page Preservation & Route Fixes | PDF physical page extraction, page-aware chunking, vector payload & `/api/v1/rag/search` route alias | none | DONE |
| M2 | Backend Knowledge Graph, Provenance & Stability Persistence | SQLite migrations (`knowledge_points`, `kp_relations`, `messages`), `LearningStore` upsert/rollback/mastery, REST APIs | none | DONE |
| M3 | Frontend Text Selection & Dual-Zone Provenance Card | Floating action menu on text selection, dual-zone knowledge card, two-way chat & document navigation | M1, M2 | DONE |
| M4 | Typed Semantic Canvas Edges & Backend Sync | `CanvasEdge` semantic types, SVG rendering, and synchronization to backend `kp_relations` & `kp_prereq` | M2, M3 | DONE |
| M5 | Message Tree Branching Follow-up & Revision History UI | Node branch follow-up via message-tree anchoring, canvas branch indicator/switcher, revision history drawer & rollback | M2, M3 | DONE |
| M6 | Final Verification & Coverage Hardening | Phase 1: 100% pass of E2E Test Suite (Tiers 1-4); Phase 2: Adversarial coverage hardening (Tier 5) | E2E, M1, M2, M3, M4, M5 | DONE |
| FIX-P0 | Truthful Provenance, Scope Isolation & Data Ownership | Features 19-22: real page attribution/preview, tool & canvas scope wiring, data ownership | M1-M6 | DONE |
| FIX-P1 | Backend-First Versions, Deletion Sync, Hydration, Branch Switching, Session Memory | Backend `history_json` as single version source, delete cascades, canvas backend hydration, canvas branch switcher, session-scoped memory consolidate/remove | FIX-P0 | DONE |
| FIX-P2 | Auth Boundary, Full-Stack E2E, Repo Hygiene | Optional `API_TOKEN`, offline full-stack Playwright suite, docs/gitignore updates | FIX-P1 | DONE |
| FIX-P3 | Temporary KB (Staging), Web Import, Distillation, Promotion, MCP Supplement | Features 25-30: SSRF-checked web fetch, staging tier isolation, TTL, distill candidates, promote/revoke into standard KB, B-channel supplement via MCP/provider | FIX-P2 | DONE |

## Interface Contracts

### Frontend ↔ Backend: Knowledge Node Upsert & Revision
- Endpoint: `POST /api/courses/{course_id}/knowledge`
  - Request:
    ```json
    {
      "name": "Knowledge Node Title",
      "summary": "Short summary",
      "difficulty": 0.5,
      "raw_quote": "Verbatim quote from chat or source document",
      "content": "User annotations or synthesized markdown",
      "provenance": {
        "message_id": "msg_xyz",
        "turn_index": 2,
        "doc_title": "Hello-Agents.pdf",
        "page_start": 42,
        "page_end": 43,
        "quote_snippet": "Selected snippet"
      },
      "prereq_ids": ["kp_prereq_1"]
    }
    ```
  - Response:
    ```json
    {
      "status": "ok",
      "id": "kp_abc123",
      "version": 1,
      "created_at": 1729340000.0
    }
    ```

### Frontend ↔ Backend: Semantic Relations Sync
- Endpoint: `POST /api/courses/{course_id}/knowledge/relations`
  - Request:
    ```json
    {
      "src_id": "kp_1",
      "rel_type": "prerequisite" | "causal" | "similar" | "branch_followup",
      "dst_id": "kp_2",
      "metadata": { "label": "custom optional label" }
    }
    ```
  - Response: `{"status": "ok", "created": true}`

### Backend RAG ↔ Frontend / LLM: Page Numbers in Retrieval
- Endpoint: `POST /api/v1/rag/search` and `POST /api/rag/search`
  - Request: `{"query": "...", "top_k": 5}`
  - Response item:
    ```json
    {
      "chunk_index": 3,
      "content": "...",
      "doc_id": "doc_xyz",
      "filename": "reference_doc.pdf",
      "page_start": 42,
      "page_end": 43,
      "headers": ["Chapter 3", "Section 3.1"]
    }
    ```

### Frontend Internal: Message Tree Branching Contract
- Actual implementation anchors the branch point by switching the conversation leaf (`conversationsStore.updateCurrentNode(anchorMessageId)` + `refreshActiveMessages()`), then sending a normal message: the new user message becomes a sibling of the original follow-up, preserving the original branch.
- Canvas branch indicator/switcher calls `conversationsStore.navigateToSibling(leafId)`; `buildSiblingInfoMap` provides `{currentIndex, totalSiblings, siblingIds}`.

### Frontend ↔ Backend: Real Provenance & Page Preview
- `POST /api/rag/attribute` — `{quote, namespace}` -> `{matched, overlap, chunk:{doc_id, chunk_id, filename, page_start, page_end}}`; only matches above the trigram-containment threshold.
- `GET /api/rag/documents/{doc_id}/file?namespace=` — original uploaded file bytes (PDF rendered by pdf.js in the browser).
- `GET /api/rag/documents/{doc_id}/page?namespace=&page=` — extracted page text for non-PDF/text documents.
- `DELETE /api/courses/{course_id}/knowledge/{kp_id}` — user-scoped physical delete that cascades relations, prerequisites, and stability rows (`mastery`).

### Isolation & Identity Headers
- `X-User-Id` (default `default_user`), `X-RAG-Namespace`, `X-Course-Id` scope state/retrieval calls.
- Tool layer (`/tools`, `/mcp`) resolves the same scope from the bound session; `rag_search_tool` no longer accepts a namespace argument.
- Optional `API_TOKEN`: when set, every route except `/health` requires `Authorization: Bearer <token>` (WebUI reuses the API Key setting).

## Verification Commands
```powershell
# Backend
.venv\Scripts\pytest.exe tests -q
# Frontend type check
cd llama-webui\tools\ui; npm run check
# Frontend unit tests
npm run test:unit -- --run
# Full-stack browser E2E (offline FakeLLM backend + Vite dev server)
npm run test:e2e:fullstack
```
