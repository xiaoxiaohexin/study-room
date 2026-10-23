# TEST_READY: 测试执行现状

> **最近更新**: 2026-10-23
> **验证环境**: Python 3.11 / Windows，`uv.lock` 锁定依赖
> **当前结果**: 全量后端测试 **508 passed, 0 failed**（`pytest -q`，约 76s）

本文件取代 2026-10-20 的发布报告（其中 "31 passed / 17 failed (pending M2)" 已过期，
那些用例在 M2 完成后即全部通过）。

---

## 1. 运行方式

```bash
# 后端全量测试
.venv/Scripts/python.exe -m pytest -q

# 只跑隔离 / 审计整改相关回归
.venv/Scripts/python.exe -m pytest -q tests/test_isolation_e2e.py tests/test_audit_regressions.py

# 前端类型检查与单测（在 llama-webui/tools/ui 下）
npm run check
npm run test:unit -- --run
```

> `pytest.ini` 已设置 `asyncio_mode = auto`，异步用例无需显式 marker。

---

## 2. 测试套件一览

| 套件 | 文件 | 覆盖范围 |
|:---|:---|:---|
| 记忆与 RAG 隔离 E2E | `tests/test_isolation_e2e.py` | 跨用户/跨会话的记忆、知识库、SM-2 掌握度隔离 |
| 审计整改回归 | `tests/test_audit_regressions.py` | 终端工具沙箱、笔记路径穿越、图片越界读取、判分兜底、`/tools` 状态码、跨站请求、题库归属、记忆幂等与持久化、依赖声明 |
| 认证与作用域 | `tests/test_auth_token.py`、`tests/test_tool_scope_isolation.py` | 可选 API Token 边界、工具检索作用域 |
| 学习域 | `tests/test_learning.py`、`tests/test_learning_isolation.py` | 意图路由、SM-2、掌握度、错题本 |
| 记忆系统 | `tests/test_memory.py`、`tests/test_memory_session_consolidation.py` | 四层记忆、整合与遗忘 |
| RAG / 入库 | `tests/test_chunker.py`、`tests/test_rag_provenance.py`、`tests/test_rag_adversarial_m1.py` | 切块、页码溯源、多格式加载 |
| Agent 循环 | `tests/test_agent_loop.py`、`tests/test_parser.py` | ReAct 多步、工具协议解析 |
| 对抗/压力 | `tests/test_challenger_*.py`、`tests/test_auditor_m3_adversarial.py`、`tests/test_m1_adversarial_stress.py` | 并发命名空间隔离、会话污染、边界输入 |
| 综合 E2E | `tests/test_e2e_suite.py`、`tests/test_api.py` | 4 Tier 端到端场景（见下方已知问题） |

---

## 3. 已知问题（待 P2 工程规范阶段处理）

1. **`tests/test_e2e_suite.py` 仍是混合型**：部分用例在本地重算字符串或模拟前端消息树，
   没有走真实 FastAPI 路由，无法证明端到端行为。应逐条改为 `AsyncClient` 打真实接口。
2. **缺少 `tests/conftest.py`**：各测试文件各自搭建 SQLite / 容器夹具，重复度高。
3. **缺少应用级 `Dockerfile` 与 GitHub Actions CI**：目前没有干净环境安装冒烟测试
   （`pip install -e . && pytest` 的自动化门禁）。
4. **大文件待拆分**：`InfiniteCanvasWorkspace.svelte`、`llama_compat.py`、`LearningStore`
   仍为单文件实现。