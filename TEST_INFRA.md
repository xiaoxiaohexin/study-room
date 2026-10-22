# Test Infrastructure Specification (TEST_INFRA)

## 1. Executive Summary & Purpose

This document establishes the architecture, test harness, methodology, and execution protocols for the End-to-End (E2E) Test Suite of the **Backend Memory & RAG Knowledge Base Layered Isolation and WebUI Binding Project** (`e:\xiaoxiao`).

The testing system is engineered according to **opaque-box, requirement-driven, zero-facade principles**:
- **Opaque-Box**: Tests interact strictly through public interfaces, REST API contracts (`/api/sessions`, `/api/chat`, `/v1/chat/completions`), SQLite schema contracts, and domain models specified in `PROJECT.md` and `ORIGINAL_REQUEST.md`. No test relies on private implementation details or mutable internal variables.
- **Deterministic Oracles**: Every assertion is derived from the project specifications, mathematical models (SM-2, session-affinity scoring multipliers, vector collection partitioning), and documented data contracts.
- **Progressive Testability & Hermetic Isolation**: Every test creates its own independent environment (ephemeral SQLite database via `tmp_path`, independent container and settings, in-memory or ephemeral vector stores, and local deterministic embedders), guaranteeing zero cross-test interference and reproducible pass/fail reporting across milestones.

---

## 2. Test Architecture & Harness

### 2.1 Backend Pytest Harness
- **Test Runner**: `pytest 9.1+` with `pytest-asyncio` and `anyio`.
- **Execution Runtime**: Python 3.11+ virtual environment (`.venv`).
- **HTTP Transport**: `httpx.AsyncClient` with `ASGITransport(app=app_instance)` for high-speed, in-process, non-network-bound ASGI API invocation.
- **Database Isolation**: Each test fixture instantiates an ephemeral `SQLiteStore` backed by a unique temporary directory (`tmp_path / "test_iso.db"`). Connections are cleanly initialized and closed via `aclose()`.
- **Vector Store Isolation**: Uses in-memory (`InMemoryVectorStore`) and temporary SQLite vector storage (`SQLiteVectorStore`), avoiding external daemon dependencies (e.g., Qdrant) during automated regression runs.
- **Embedding Oracle**: `HashingEmbedder(dim=32 or 128)` for instantaneous, deterministic, non-network-dependent vector operations.
- **Container Lifecycle**: `AppContainer(settings)` is initialized per fixture and explicitly closed during teardown, releasing all file handles and connection pools.

### 2.2 Frontend Vitest & Playwright Harness
- **Unit & Contract Runner**: `vitest 4.1+` configured with `@sveltejs/kit/vite` and client/unit projects in `llama-webui/tools/ui`.
- **Type & Compile Check**: `svelte-check --tsconfig ./tsconfig.json` ensuring zero type drift in `InfiniteCanvasWorkspace.svelte`, `ChatScreen.svelte`, and store preference bindings.
- **Contract Tests**: `tests/unit/session-rag-namespace.test.ts` verifying `DatabaseConversation` persistence, header definitions, and `ChatService` transmission.

### 2.3 Directory Layout
```
e:\xiaoxiao\
├── tests\
│   ├── test_isolation_e2e.py       # Primary 4-Tier E2E automated test suite for Memory & RAG Isolation
│   ├── test_e2e_suite.py           # Knowledge canvas & citation baseline test suite (177 tests)
│   ├── test_challenger_m1.py       # Milestone 1 adversarial stress suite
│   ├── test_challenger_m2.py       # Milestone 2 relation & mastery stress suite
│   ├── test_api.py                 # Core API endpoints & LlamaCompat
│   ├── test_learning.py            # Learning domain models, SM-2, Mastery
│   ├── test_chunker.py             # MarkdownChunker structure & headers
│   └── test_memory.py              # Baseline memory lifecycle tests
├── llama-webui\tools\ui\tests\
│   ├── unit\
│   │   ├── session-rag-namespace.test.ts # WebUI session KB binding & preference contracts
│   │   └── branching.test.ts             # Message tree branching algorithms
│   ├── client\                     # Component-level tests
│   └── e2e\                        # Playwright browser flows
├── TEST_INFRA.md                   # This specification file
└── TEST_READY.md                   # Execution report, pass/fail counts, coverage matrix
```

---

## 3. Systematic 4-Tier Methodology

The suite covers all features from `PROJECT.md` across four distinct verification tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Tier 4: Real-World Workflows (5 Scenarios)           │
│   Multi-Course Study Journeys, Coding Scratchpad & Lifecycle Closures  │
├────────────────────────────────────────────────────────────────────────┤
│             Tier 3: Cross-Feature Pairwise Integrations (5 Combos)     │
│   Session Switching, Global Mastery Sync, Dynamic RAG Re-binding       │
├────────────────────────────────────────────────────────────────────────┤
│         Tier 2: Boundary, Extreme & Negative Error Conditions          │
│   (>=8 Tests: Unsessioned Fallback, Nonexistent Namespaces, Unicode)   │
├────────────────────────────────────────────────────────────────────────┤
│                Tier 1: Feature Isolation Coverage                      │
│   (>=5 Tests / Feature Area across F_MEM, F_EPISODIC, F_SEMANTIC,      │
│    F_DB, F_RAG, F_API = 30 Test Cases)                                 │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Feature Inventory Mapping (Features F_MEM - F_API)

| # | Feature Area | Identifier | Core Contract / Target Component |
|---|--------------|------------|----------------------------------|
| 1 | Working Memory Isolation | `F_MEM` | Compound storage key `(user_id, session_id)` in `WorkingMemory._store` |
| 2 | Episodic Memory Prioritization | `F_EPISODIC` | Session affinity boost (1.35x), cross-session penalty (0.65x), and <0.50 cutoff |
| 3 | Global Semantic & SM-2 Profile | `F_SEMANTIC` | User-scoped global mastery & concept graph preservation across all sessions |
| 4 | SQLite Schema & Migration | `F_DB` | Idempotent migration of `sessions` (`course_id`, `rag_namespace`) |
| 5 | RAG Namespace Physical Isolation | `F_RAG` | Physical collection separation `rag_{namespace}_{embedder_tag}` |
| 6 | Session CRUD REST APIs | `F_API` | `POST /api/sessions`, `GET /api/sessions/{u}/{s}`, `PATCH`, `DELETE` |

---

## 4. Test Suite Specification by Tier

### Tier 1: Feature Isolation Coverage (30 Tests)
Provides >=5 isolated tests for each of the 6 core feature areas:
- **F_MEM (Working Memory Isolation)**:
  - `test_f_mem_01_compound_key_isolation`: Compound key `(user_id, session_id)` strictly isolates session items.
  - `test_f_mem_02_tool_scratchpad_unpolluted_across_sessions`: Tool scratchpads from Session 1 do not leak to Session 2.
  - `test_f_mem_03_list_items_scoped_to_session`: `list_items` returns only the memories belonging to target session.
  - `test_f_mem_04_session_capacity_independence`: Filling capacity in Session A does not evict items from Session B.
  - `test_f_mem_05_memory_manager_retrieve_all_passes_session_id`: `MemoryManager.retrieve_all` forwards `session_id`.
- **F_EPISODIC (Episodic Prioritization & Filtering)**:
  - `test_f_episodic_01_current_session_affinity_boost`: Current session affinity boost (1.35x) prioritizes current session items.
  - `test_f_episodic_02_cross_discipline_weak_relevance_filtered`: Cross-discipline low relevance (<0.50) candidates are filtered out.
  - `test_f_episodic_03_vector_payload_stores_session_id`: Vector points and SQLite rows persist `session_id`.
  - `test_f_episodic_04_same_user_multi_session_history_separation`: Distinct course queries do not cross-contaminate.
  - `test_f_episodic_05_memory_lifecycle_consolidate_scoped_by_session`: Working memory consolidation is session-scoped.
- **F_SEMANTIC (Semantic Memory & SM-2 Mastery)**:
  - `test_f_semantic_01_global_user_scope_across_sessions`: Semantic concepts are retrievable globally across all sessions.
  - `test_f_semantic_02_sm2_mastery_persists_globally_across_sessions`: Mastery updated in Session 1 is visible in Session 2.
  - `test_f_semantic_03_learner_profile_text_aggregates_across_sessions`: Learner profile aggregates global mistakes and progress.
  - `test_f_semantic_04_different_users_mastery_isolated`: Different users' mastery states are strictly partitioned.
  - `test_f_semantic_05_new_session_inherits_global_knowledge_graph`: New sessions immediately inherit user concept graphs.
- **F_DB (SQLite Schema Migration & Session Model)**:
  - `test_f_db_01_sessions_table_has_course_id_and_rag_namespace`: Inspect `sessions` table PRAGMA columns.
  - `test_f_db_02_idempotent_migration_on_preexisting_sessions_table`: Pre-existing database upgrade without data loss.
  - `test_f_db_03_sessions_table_indexes_exist`: Indices `idx_sessions_course` and `idx_sessions_rag_ns` exist.
  - `test_f_db_04_default_rag_namespace_value`: Unspecified session namespace defaults to `'default'`.
  - `test_f_db_05_session_state_manager_persists_and_reads_bindings`: `SessionStateManager` CRUD read/write of bindings.
- **F_RAG (RAG Physical Isolation & Resolution)**:
  - `test_f_rag_01_physical_collection_partitioning`: Physics and Literature collections are physically partitioned.
  - `test_f_rag_02_collection_name_standard_format`: Collection naming matches `rag_{namespace}_{embedder.tag}`.
  - `test_f_rag_03_empty_namespace_returns_zero_results`: Empty namespace returns 0 hits, zero leakage from default.
  - `test_f_rag_04_api_header_x_rag_namespace_routing`: Header `X-RAG-Namespace` routes retrieval to target space.
  - `test_f_rag_05_api_body_rag_namespace_routing`: Body parameter `rag_namespace` routes retrieval to target space.
- **F_API (Session CRUD REST API)**:
  - `test_f_api_01_post_create_session_with_bindings`: `POST /api/sessions` creates session with course/namespace bindings.
  - `test_f_api_02_get_session_detail_returns_bindings`: `GET /api/sessions/{u}/{s}` returns details and bindings.
  - `test_f_api_03_patch_session_updates_bindings`: `PATCH /api/sessions/{u}/{s}` updates course/namespace/title.
  - `test_f_api_04_get_user_sessions_list_includes_bindings`: `GET /api/sessions/{u}` lists all sessions with bindings.
  - `test_f_api_05_delete_session`: `DELETE /api/sessions/{u}/{s}` removes session cleanly.

### Tier 2: Boundary, Extreme & Negative Conditions (8 Tests)
- `test_boundary_01_un_sessioned_working_memory_fallback`: Legacy un-sessioned queries fall back safely.
- `test_boundary_02_empty_string_session_id_handling`: `session_id=""` does not cause collision or crash.
- `test_boundary_03_nonexistent_namespace_zero_leakage`: Querying ghost namespace yields 0 hits, 0 leaks.
- `test_boundary_04_unicode_and_special_character_namespaces`: Special characters (`高等数学·微积分_v2.0-2026/Alpha`) supported.
- `test_boundary_05_long_ids_stress`: 128-char `user_id` and `session_id` handled without truncation.
- `test_boundary_06_preexisting_db_migration_preserves_50_sessions`: Migration preserves 50 legacy rows with `'default'`.
- `test_boundary_07_get_nonexistent_session_404`: Nonexistent session returns 404.
- `test_boundary_08_patch_nonexistent_session_404`: Patching nonexistent session returns 404.

### Tier 3: Cross-Feature Pairwise Integrations (5 Tests)
- `test_pairwise_01_multi_session_switch_zero_bleed`: Active switching between Physics and Literature sessions exhibits zero memory or RAG chunk bleed.
- `test_pairwise_02_global_mastery_sync_across_isolated_course_sessions`: Mistake recorded in Math session updates global SM-2 mastery, visible in Physics session while scratchpads remain isolated.
- `test_pairwise_03_session_binding_update_dynamically_redirects_rag`: `PATCH` session to `ns_v2` dynamically redirects future RAG retrievals without restarting.
- `test_pairwise_04_concurrent_sessions_same_user`: Concurrent requests across 2 sessions maintain independent states race-free.
- `test_pairwise_05_precedence_hierarchy_header_over_session`: Header override takes precedence over session binding.

### Tier 4: Real-World Multi-Course Study Workloads (5 Scenarios)
- `test_workload_01_multicourse_student_semester_journey`: Ingest 3 textbooks -> Bind 3 sessions -> Multi-turn retrieval -> Verify citations match physical pages.
- `test_workload_02_tool_execution_scratchpad_isolation_during_coding`: Python coding tool scratchpad in Session A never enters Session B.
- `test_workload_03_cross_disciplinary_distraction_defense`: Chit-chat in recreational session is filtered out when studying in calculus session.
- `test_workload_04_sm2_spaced_repetition_unified_across_study_sessions`: SM-2 review ratings across 3 independent sessions advance global interval progression.
- `test_workload_05_full_lifecycle_session_crud_and_cleanup`: Complete lifecycle: create session -> query -> patch -> list -> delete -> verify clean DB state.

---

## 5. Execution Commands

```powershell
# 1. Run complete Memory & RAG Namespace Isolation E2E test suite (48 tests)
.\.venv\Scripts\pytest.exe tests/test_isolation_e2e.py --basetemp=.pytest_temp_e2e -v

# 2. Run specific Tiers
.\.venv\Scripts\pytest.exe tests/test_isolation_e2e.py -k "TestTier1" --basetemp=.pytest_temp_e2e
.\.venv\Scripts\pytest.exe tests/test_isolation_e2e.py -k "TestTier2" --basetemp=.pytest_temp_e2e
.\.venv\Scripts\pytest.exe tests/test_isolation_e2e.py -k "TestTier3" --basetemp=.pytest_temp_e2e
.\.venv\Scripts\pytest.exe tests/test_isolation_e2e.py -k "TestTier4" --basetemp=.pytest_temp_e2e

# 3. Run frontend contract unit tests (Vitest)
cd llama-webui\tools\ui
npm.cmd run test:unit -- --run tests/unit/session-rag-namespace.test.ts

# 4. Run frontend typecheck
npm.cmd run check

# 5. Run full baseline regression suite
.\.venv\Scripts\pytest.exe tests/test_learning.py tests/test_chunker.py tests/test_api.py --basetemp=.pytest_temp_e2e -q
```

---

## 6. Progressive Milestone Pass Criteria

| Milestone | Target Scope | Expected E2E Suite Impact | Current Status |
|:---|:---|:---|:---:|
| **E2E Baseline (Current)** | Test suite published; Database schema, RAG physical isolation, API CRUD, and Frontend contracts validated | 31 PASSED, 17 FAILED (deterministic pending M2 memory isolation) | **PUBLISHED** |
| **M1 Completion** | SQLite Migration, Session Model & CRUD API routes | `test_f_db_*`, `test_f_api_*` fully green | **READY** |
| **M2 Completion** | Working Memory per-(user_id, session_id), Episodic session-prioritized scoring | `test_f_mem_*`, `test_f_episodic_*`, `test_pairwise_*`, `test_workload_*` flip to PASS (17 tests) | **PENDING M2** |
| **M3 Completion** | API Routes Header Extraction & Context Retrieval Isolation | RAG routing and LlamaCompat pre-fetch verified | **PENDING M3** |
| **M4 Completion** | Frontend WebUI Session KB Selection & Passthrough | Vitest unit tests (8 tests) and svelte-check pass | **READY** |
| **M5 Completion** | Final Verification & Zero Regression Hardening | **100% of all 48 isolation tests and 266+ existing backend tests PASS** | **PENDING M5** |
