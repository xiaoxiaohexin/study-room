# Project: Knowledge Canvas & Conversation/Textbook Deep Integration

## Architecture
- **Backend RAG & Document Processing**: `app/rag/loader.py`, `chunker.py`, `pipeline.py`, `retriever.py`. Ingests textbooks (PDF/Markdown), preserves exact physical page numbers (`page_start`/`page_end`) and section hierarchy, vectorizes and stores with page-aware metadata.
- **Backend Storage & Knowledge Graph**: `app/storage/sqlite.py`, `app/learning/store.py`, `app/learning/graph.py`. Manages SQLite schemas: `knowledge_points` (provenance, raw excerpt, versions), `kp_relations` (typed semantic edges: `prerequisite`, `causal`, `similar`, `branch_followup`), `kp_prereq`, `mastery`, and `messages` (`parent_id` message tree).
- **Backend REST API**: `app/api/routes/rag.py`, `learn.py`, `llama_compat.py`. Exposes `/api/v1/rag/search` alias, `/api/courses/{course_id}/knowledge` endpoints (CRUD, revisions, rollback, relations sync), and normalized tool execution.
- **Frontend Chat & Selection**: `llama-webui/tools/ui/src/lib/components/app/chat/ChatMessages/`. Listens to text selection on assistant responses, displays floating toolbar with "提取为知识节点" / "追加到已有节点".
- **Frontend Canvas Workspace**: `InfiniteCanvasWorkspace.svelte`. Renders dual-zone cards (`kind: 'knowledge'`, immutable quote excerpt zone + editable user note zone), provenance badges with two-way click navigation (chat message scroll/highlight + PDF page preview), typed semantic edges (`CanvasEdge`), and version history drawer.
- **Frontend Message Tree & Branching**: `database.service.ts`, `branching.ts`, `chatStore`, `conversationsStore`. Implements true sub-branch question follow-up from canvas nodes using `pendingBranchParentId`, branching tree visualization, and lossless branch switching.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | RAG Physical Page Extraction | `DocumentLoader` extracts PDF pages using `pypdf` with `<!-- PAGE: N -->` delimiters | M1 | ORIGINAL_REQUEST §R2 |
| 2 | Page-Aware Chunking | `MarkdownChunker` detects page markers, records `page_start` & `page_end` on chunks | M1 | ORIGINAL_REQUEST §R2 |
| 3 | Page-Aware Vector Payload & Citation | `IngestionPipeline` & `RagRetriever` return `page_start`/`page_end` in search and prompt citations (`P.42`) | M1 | ORIGINAL_REQUEST §R2 |
| 4 | RAG API Routing Alias | Support `/api/v1/rag/search` routing alias matching frontend expectations | M1 | Survey Backend |
| 5 | Knowledge Point Schema Migration | Idempotently migrate `knowledge_points` table with `raw_quote`, `content`, `version`, `provenance_json`, `history_json` | M2 | ORIGINAL_REQUEST §R1, R5 |
| 6 | Semantic Relations Schema (`kp_relations`) | SQLite table `kp_relations` supporting `prerequisite`, `causal`, `similar`, `branch_followup` | M2 | ORIGINAL_REQUEST §R3 |
| 7 | Dual-Write Prerequisite Sync | Dual-write `prerequisite` relations to `kp_prereq` and `GraphStore` for backward compatibility | M2 | ORIGINAL_REQUEST §R3 |
| 8 | Mastery Enrollment & Knowledge APIs | `upsert_knowledge_point` auto-enrolls into `mastery`, handles provenance, revisions, and rollback REST endpoints | M2 | ORIGINAL_REQUEST §R1, R5 |
| 9 | Text Selection Floating Toolbar | Text selection in chat assistant messages triggers floating action menu to extract/append knowledge cards | M3 | ORIGINAL_REQUEST §R1 |
| 10 | Dual-Zone Knowledge Canvas Card | Card with immutable read-only excerpt zone and editable user content zone, displaying provenance badges | M3 | ORIGINAL_REQUEST §R1, R2, R5 |
| 11 | Bi-directional Chat & Textbook Anchors | Clicking chat badge scrolls & highlights message; clicking textbook badge previews PDF page | M3 | ORIGINAL_REQUEST §R2 |
| 12 | Strong Semantic Canvas Edges | Upgraded `CanvasEdge` with distinct SVG styles and menu for `prerequisite`, `causal`, `similar`, `branch_followup` | M4 | ORIGINAL_REQUEST §R3 |
| 13 | Edge Persistence & SQLite Sync | Syncing canvas semantic edges with backend `kp_relations` and `kp_prereq` | M4 | ORIGINAL_REQUEST §R3 |
| 14 | Node Branch Follow-up Engine | Branching follow-up from canvas node setting `pendingBranchParentId` to origin message ID in Dexie tree | M5 | ORIGINAL_REQUEST §R4 |
| 15 | Canvas Branch Indicator & Lossless Switching | Canvas turn cards show a branch indicator/switcher and switch branches via `conversationsStore.navigateToSibling` (full branch-tree graph view deferred) | M5 + Fix P1 | ORIGINAL_REQUEST §R4 |
| 16 | Revision History & Non-Destructive Rollback | Card revision badge `v{N} · 历史`, history timeline modal, and one-click rollback | M5 | ORIGINAL_REQUEST §R5 |
| 17 | E2E Test Suite (Tiers 1-4) | Comprehensive automated test suite covering all 16 features across Tier 1 (Feature), Tier 2 (Boundary), Tier 3 (Cross-feature), Tier 4 (Workload) | E2E-Track / M6 | ORIGINAL_REQUEST §Verification |
| 18 | Adversarial Coverage Hardening (Tier 5) | White-box adversarial testing and edge case hardening | M6 | Project Pattern |
| 19 | Learning Data Ownership Isolation | `knowledge_points.user_id` + course auto-registration, user/course scoped CRUD, cross-course write/lookup rejected, relation endpoint consistency checks | Fix P0-A | Review findings |
| 20 | Tool-Layer Scope Isolation | `ToolContext.rag_namespace/course_id` propagated from session; RAG/quiz/knowledge tools cannot select another namespace or user scope | Fix P0-B | Review findings |
| 21 | Real Provenance & Page Preview | RAG document registry, `/api/rag/documents/{id}/file|page`, quote attribution (`/api/rag/attribute`), real PDF page rendering via pdf.js (no fabricated "参考教材 P.1") | Fix P0-C/D | Review findings |
| 22 | Frontend Scope Wiring | `learning-context` single source for course/namespace/user, real turn indices, canvas as single knowledge writer, backend hydration, backend-first versions | Fix P0-D/P1 | Review findings |
| 23 | Optional API Token Boundary | `API_TOKEN` opt-in bearer auth for all endpoints except `/health`; WebUI reuses API Key header | Fix P2 | Review findings |
| 24 | Full-Stack Browser E2E | Offline FakeLLM backend + real Vite dev server Playwright suite (selection→card→anchor, branch switching, real PDF page preview, web import→distill→promote) | Fix P2/P3 | Review findings |
| 25 | Web Import (A channel) | URL/search discovery → SSRF-checked fetch → readability extraction → staged snapshot in temporary KB | FIX-P3 | Staging request |
| 26 | Temporary Knowledge Base (Staging) | `documents.tier/source_type/url/content_hash`, isolated `staging_{embedder}` collection, per-session dedupe/capacity, 7-day TTL cleanup, never part of standard RAG | FIX-P3 | Staging request |
| 27 | Distillation to Knowledge Candidates | Quality score + LLM structured extraction (title/summary/content/raw_quote) with rule fallback → pending candidate inbox | FIX-P3 | Staging request |
| 28 | Promotion & Revoke | Promote staged snapshot + candidates into target course namespace and knowledge points (canvas hydrates them); one-click revoke removes standard doc + created KPs | FIX-P3 | Staging request |
| 29 | Runtime Web Supplement (B channel) | RAG absolute-similarity gate → MCP search / provider → fetch + stage + `[WEB]` prompt block with URL citations (auto mode, off by default) | FIX-P3 | Staging request |
| 30 | WebTool over MCP | `web_tool` in ToolRegistry therefore exposed via `/tools` and `/mcp` (search/fetch/stage) with session scope | FIX-P3 | Staging request |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite Track | Complete opaque-box automated test suite (Tiers 1-4, 177 tests), TEST_INFRA.md & TEST_READY.md | none | DONE |
| M1 | Backend RAG Physical Page Preservation & Route Fixes | PDF physical page extraction, page-aware chunking, vector payload & `/api/v1/rag/search` route alias | none | DONE |
| M2 | Backend Knowledge Graph, Provenance & Mastery Persistence | SQLite migrations (`knowledge_points`, `kp_relations`, `messages`), `LearningStore` upsert/rollback/mastery, REST APIs | none | DONE |
| M3 | Frontend Text Selection & Dual-Zone Provenance Card | Floating action menu on text selection, dual-zone knowledge card, two-way chat & textbook navigation | M1, M2 | DONE |
| M4 | Typed Semantic Canvas Edges & Backend Sync | `CanvasEdge` semantic types, SVG rendering, and synchronization to backend `kp_relations` & `kp_prereq` | M2, M3 | DONE |
| M5 | Message Tree Branching Follow-up & Revision History UI | Node branch follow-up via message-tree anchoring, canvas branch indicator/switcher, revision history drawer & rollback | M2, M3 | DONE |
| M6 | Final Verification & Coverage Hardening | Phase 1: 100% pass of E2E Test Suite (Tiers 1-4); Phase 2: Adversarial coverage hardening (Tier 5) | E2E, M1, M2, M3, M4, M5 | DONE |
| FIX-P0 | Truthful Provenance, Scope Isolation & Learning Ownership | Features 19-22: real page attribution/preview, tool & canvas scope wiring, learning data ownership | M1-M6 | DONE |
| FIX-P1 | Backend-First Versions, Deletion Sync, Hydration, Branch Switching, Session Memory | Backend `history_json` as single version source, delete cascades, canvas backend hydration, canvas branch switcher, session-scoped memory consolidate/remove | FIX-P0 | DONE |
| FIX-P2 | Auth Boundary, Full-Stack E2E, Repo Hygiene | Optional `API_TOKEN`, offline full-stack Playwright suite, docs/gitignore updates | FIX-P1 | DONE |
| FIX-P3 | Temporary KB (Staging), Web Import, Distillation, Promotion, MCP Supplement | Features 25-30: SSRF-checked web fetch, staging tier isolation, TTL, distill candidates, promote/revoke into standard KB, B-channel supplement via MCP/provider | FIX-P2 | DONE |

## Interface Contracts

### Frontend ↔ Backend: Knowledge Point Upsert & Revision
- Endpoint: `POST /api/courses/{course_id}/knowledge`
  - Request:
    ```json
    {
      "name": "Knowledge Point Title",
      "summary": "Short summary",
      "difficulty": 0.5,
      "raw_quote": "Verbatim quote from chat or textbook",
      "content": "User notes or synthesized markdown",
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
      "filename": "textbook.pdf",
      "page_start": 42,
      "page_end": 43,
      "headers": ["Chapter 3", "Section 3.1"]
    }
    ```

### Frontend Internal: Message Tree Branching Contract
- Actual implementation anchors the branch point by switching the conversation leaf (`conversationsStore.updateCurrentNode(anchorMessageId)` + `refreshActiveMessages()`), then sending a normal message: the new user message becomes a sibling of the original follow-up, preserving the original branch.
- Canvas branch indicator/switcher calls `conversationsStore.navigateToSibling(leafId)`; `buildSiblingInfoMap` provides `{currentIndex, totalSiblings, siblingIds}`.
- Deferred: a full branch-tree graph visualization on the canvas (only the indicator + previous/next switching is implemented).

### Frontend ↔ Backend: Real Provenance & Page Preview
- `POST /api/rag/attribute` — `{quote, namespace}` -> `{matched, overlap, chunk:{doc_id, chunk_id, filename, page_start, page_end}}`; only matches above the trigram-containment threshold.
- `GET /api/rag/documents/{doc_id}/file?namespace=` — original uploaded file bytes (PDF rendered by pdf.js in the browser).
- `GET /api/rag/documents/{doc_id}/page?namespace=&page=` — extracted page text for non-PDF/text documents.
- `DELETE /api/courses/{course_id}/knowledge/{kp_id}` — cascades relations, prerequisites and mastery rows.

### Isolation & Identity Headers
- `X-User-Id` (default `default_user`), `X-RAG-Namespace`, `X-Course-Id` scope learning/retrieval calls.
- Tool layer (`/tools`, `/mcp`) resolves the same scope from the bound session; `rag_search_tool` no longer accepts a namespace argument.
- Optional `API_TOKEN`: when set, every route except `/health` requires `Authorization: Bearer <token>` (WebUI reuses the API Key setting).

### Web / Temporary KB (Staging) Contracts
- `GET /api/web/status` — enabled / search availability / TTL.
- `POST /api/web/preview` — fetch + extract preview (no persistence).
- `POST /api/web/stage` — stage a URL into the temporary KB (web tier).
- `POST /api/web/search` — search discovery; `stage=true` also fetches & stages top results. Provider chain: SearXNG -> Tavily OR built-in Bing RSS fallback (`WEB_SEARCH_FALLBACK_PROVIDER=bing`, no install needed); each failure is reported with an actionable hint.
- `GET/DELETE /api/staging/documents` — list/discard staged documents (never returned by `/api/rag/documents`).
- `POST /api/staging/documents/{id}/distill` — distill into pending knowledge candidates.
- `GET /api/staging/candidates`, `POST /api/staging/candidates/{id}/reject`.
- `POST /api/staging/promote` — promote staged docs + candidates into `target_course_id` namespace & create knowledge points.
- `POST /api/staging/revoke` — undo a promotion (delete standard doc + created knowledge points).
- B channel: `RagRetriever.confidence()` gate + `WebSupplementService`; only `WEB_FALLBACK_MODE=auto` enables it, failures degrade to local-only.
- MCP: `web_tool` (`search`/`fetch`, optional auto-stage) is registered in ToolRegistry and therefore listed by both `/tools` and `/mcp`; third-party MCP search servers can be used via `MCP_WEB_SERVER_URL`.

## Code Layout
- Backend Source:
  - `app/rag/loader.py` — `DocumentLoader` with `pypdf` page marker extraction
  - `app/rag/chunker.py` — `MarkdownChunker` with page delimiter tracking
  - `app/rag/pipeline.py` — `IngestionPipeline` storing `page_start`/`page_end`
  - `app/rag/retriever.py` — `RagRetriever` formatting page citations in prompt & results
  - `app/storage/sqlite.py` — SQLite schema migrations (`knowledge_points`, `kp_relations`, `messages`)
  - `app/learning/store.py` — `LearningStore` with provenance, versions, mastery, and relations sync
  - `app/api/routes/rag.py` — RAG endpoints and `/api/v1/rag/search` alias
  - `app/api/routes/learn.py` — Knowledge point CRUD, relations, revisions, and rollback REST endpoints
  - `app/api/routes/llama_compat.py` — Tool argument mapping normalization + B-channel web supplement gate
  - `app/api/routes/web.py` — Web import / staging / distill / promote / revoke APIs
  - `app/web/` — `fetcher.py` (SSRF-checked fetch + extraction), `search.py` (SearXNG/Tavily/Bing RSS + provider fallback chain), `mcp_search.py` (minimal MCP Streamable-HTTP client), `staging.py`, `distill.py`, `promotion.py`, `supplement.py`
  - `app/tools/builtin/web_tool.py` — `web_tool` exposed via `/tools` & `/mcp`
- Frontend Source:
  - `llama-webui/tools/ui/src/lib/components/app/chat/ChatMessages/ChatMessages.svelte` — Selection toolbar
  - `llama-webui/tools/ui/src/lib/components/app/chat/ChatScreen/InfiniteCanvasWorkspace.svelte` — Knowledge card, dual-zone, badges, navigation, semantic edges, revision history drawer
  - `llama-webui/tools/ui/src/lib/components/app/chat/ChatScreen/ChatScreen.svelte` — Split view, branch synchronization
  - `llama-webui/tools/ui/src/lib/stores/chat/index.svelte.ts` — Branch follow-up state (`pendingBranchParentId`)
  - `llama-webui/tools/ui/src/lib/services/database.service.ts` — IndexedDB message tree branching
- Test Suites:
  - `tests/test_learning.py`, `tests/test_chunker.py`, `tests/test_api.py` — Backend pytest
  - `tests/test_learning_isolation.py`, `tests/test_tool_scope_isolation.py`, `tests/test_rag_provenance.py`, `tests/test_memory_session_consolidation.py`, `tests/test_auth_token.py` — Fix-round isolation/provenance/auth suites
  - `tests/test_web_staging.py` — Staging tier isolation / dedupe / TTL / distill / promote / revoke / SSRF / MCP parsing / supplement gate
  - `tests/test_e2e_suite.py`, `tests/test_isolation_e2e.py` — Backend integration/isolation suites
  - `llama-webui/tools/ui/tests/unit/` — Frontend unit tests (`canvas-sync`, `learning-context`, `rag-attribution`, `turn-index`, provenance suites)
  - `llama-webui/tools/ui/tests/e2e-fullstack/` — Real browser + real backend Playwright suite (`npm run test:e2e:fullstack`)

## Known Limitations / Deferred
- Canvas layout/edges are still persisted in browser localStorage; the backend is the source of truth for knowledge content, versions, and relations (canvas hydrates missing cards on load, but layout is per-browser).
- Full canvas branch-tree graph visualization is deferred; branch indicator + previous/next switching is implemented.
- Knowledge points are still not injected into the LLM prompt (only mastery profile + RAG + memories); canvas knowledge influencing answers via prompt digest is deferred.
- Web/staging is **off by default**: `WEB_ENABLED=false`. Staging documents are intentionally excluded from standard RAG and from `/api/rag/documents`; they only participate after explicit promotion. Web content is treated as untrusted and labelled `[WEB]` with URL + fetched_at.
- The B-channel auto supplement currently runs on the llama.cpp-compatible `/v1/chat/completions` path (WebUI); the AgentLoop `/api/chat` path does not auto-supplement yet (it can use `web_tool` explicitly).
- Web search discovery needs a provider: SearXNG (self-hosted, JSON API enabled), Tavily (API key), the built-in Bing RSS fallback (`WEB_SEARCH_FALLBACK_PROVIDER=bing`, on by default), or an external MCP search server. URL import (A channel) never needs a search service. The project ships an MCP **client** only; `MCP_WEB_SERVER_URL` must point at a separately deployed MCP server.
- Non-PDF documents have no physical page numbers; the preview modal shows extracted page text and labels it as such.
- Multi-user support is API-level (caller-declared `X-User-Id` with server-side ownership checks); there is no login system. Course rows are single-owner: multi-user clients must use distinct course ids (the WebUI preset ids are shared labels, so clones should not share them across users). Keep the backend bound to 127.0.0.1 or configure `API_TOKEN` + restrictive CORS when exposing it.
- After upgrading an existing database, restart the backend once: startup migration adds `knowledge_points.user_id` / `kp_relations.user_id` and backfills the RAG `documents` registry from existing `vector_points` so old ingested PDFs become page-previewable.
- Real Qdrant/Neo4j/llama-server were unavailable during this fix round; verification used the SQLite/in-memory/offline fallbacks plus a real FastAPI + real browser E2E run.

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
