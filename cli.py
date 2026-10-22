"""智能体助手交互式简易命令行（CLI）：
支持直接本地运行或连接已启动的 FastAPI 服务，以实时时间戳和色彩化日志流式展示 Thought-Action-Observation 过程。
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# 在 Windows 终端下强制 UTF-8 编码，防止 emoji 或中文字符报 GBK 编码错误
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ANSI 色彩常量
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"


def ts() -> str:
    return f"{DIM}[{datetime.now().strftime('%H:%M:%S')}]{RESET}"


def print_banner() -> None:
    print(f"""
{CYAN}{BOLD}================================================================
       🎓 辅助学习智能体交互式终端系统 (Learning Agent CLI)
================================================================{RESET}
• 直接输入问题与导师对话（自动路由：概念精讲 / 出题 / 判卷 / 错题复习 / 计划）
• 输入 {YELLOW}/course [课程名]{RESET}       查看课程列表或新建学习课程
• 输入 {YELLOW}/kp [名称 | 简介]{RESET}      查看知识点掌握度图谱或新增知识点
• 输入 {YELLOW}/quiz <主题> [题型]{RESET}    快速生成一道针对该主题的测验题
• 输入 {YELLOW}/answer <你的答案>{RESET}     提交当前待答测验题答案并触发 SM-2 判卷
• 输入 {YELLOW}/review{RESET}                查看今日到期复习知识点 (SM-2) 与错题本
• 输入 {YELLOW}/progress{RESET}              查看个人学情看板与薄弱知识点分布
• 输入 {YELLOW}/plan [学习目标]{RESET}       查看或生成阶段性学习计划
• 输入 {YELLOW}/image <图片路径> <问题>{RESET} 进行多模态识图答疑 / 拍题批改
• 输入 {YELLOW}/ingest <文件路径>{RESET}     导入本地讲义/教材/图片到 RAG 知识库
• 输入 {YELLOW}/tools{RESET} / {YELLOW}/memory{RESET}       查看已注册工具或当前记忆库
• 输入 {YELLOW}/new{RESET}                   开启新会话
• 输入 {YELLOW}/exit{RESET} 或 {YELLOW}/quit{RESET}         退出程序
""")


async def run_local_cli(user_id: str):
    from app.core.config import get_settings
    from app.core.container import AppContainer
    from app.schemas.events import EventType
    from app.schemas.requests import ChatRequest
    from app.tools.base import ToolContext

    settings = get_settings()
    print(f"{ts()} {BLUE}正在初始化后端容器（带自动降级与状态检查）...{RESET}")
    container = AppContainer(settings)
    await container.initialize()

    print(f"{ts()} {GREEN}后端引擎初始化完成！组件健康状态：{RESET}")
    for comp, st in container.status.items():
        st_color = GREEN if st.startswith("ok") else YELLOW
        print(f"    • {comp.ljust(16)}: {st_color}{st}{RESET}")

    session = await container.state_manager.get_or_create_session(user_id=user_id)
    print(f"{ts()} {CYAN}已创建或载入会话: {session.session_id} (用户: {user_id}){RESET}\n")

    try:
        while True:
            try:
                raw_input = input(f"{BOLD}User ({user_id}) > {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break

            if not raw_input:
                continue

            if raw_input in ("/exit", "/quit"):
                print("Goodbye!")
                break

            if raw_input == "/help":
                print_banner()
                continue

            if raw_input == "/new":
                session = await container.state_manager.get_or_create_session(user_id=user_id, session_id=None)
                print(f"{ts()} {CYAN}已开启新会话: {session.session_id}{RESET}")
                continue

            if raw_input in ("/tools", "/tool"):
                specs = container.tool_registry.specs()
                print(f"\n{BOLD}已注册工具列表 ({len(specs)} 个):{RESET}")
                for s in specs:
                    print(f"  • {GREEN}{s.name}{RESET}: {s.description}")
                print()
                continue

            if raw_input == "/memory":
                memories = await container.memory_manager.list_items(user_id=user_id, limit=20)
                print(f"\n{BOLD}用户 {user_id} 当前记忆 ({len(memories)} 条):{RESET}")
                for m in memories:
                    print(f"  [{m.type.upper()}] (重要度: {m.importance:.2f}) {m.content}")
                print()
                continue

            if raw_input.startswith("/course"):
                arg = raw_input[len("/course") :].strip()
                if not arg:
                    courses = await container.learning_store.list_courses(user_id=user_id)
                    print(f"\n{BOLD}📚 课程列表 ({len(courses)} 门):{RESET}")
                    for c in courses:
                        print(f"  • [{CYAN}{c['id']}{RESET}] {BOLD}{c['name']}{RESET} (命名空间:{c.get('namespace')}) - {c.get('description') or ''}")
                    print()
                else:
                    c = await container.learning_store.create_course(user_id=user_id, name=arg)
                    print(f"{ts()} {GREEN}✅ 已创建课程: [{c['id']}] {c['name']}{RESET}\n")
                continue

            if raw_input.startswith("/kp"):
                arg = raw_input[len("/kp") :].strip()
                if not arg:
                    kps = await container.learning_store.list_knowledge_points()
                    rep = await container.learning_store.get_progress_report(user_id=user_id)
                    m_map = {m["kp_id"]: m for m in rep["knowledge_mastery"]}
                    print(f"\n{BOLD}🧠 知识点与掌握度图谱 ({len(kps)} 个):{RESET}")
                    for k in kps:
                        m = m_map.get(k["id"])
                        m_str = (
                            f"掌握度: {m['mastery']*100:.0f}% (轮次:{m['repetitions']}, 间隔:{m['interval_days']}天)"
                            if m
                            else "未练习"
                        )
                        print(f"  • [{CYAN}{k['id']}{RESET}] {BOLD}{k['name']}{RESET} (难度:{k['difficulty']}) | {YELLOW}{m_str}{RESET}")
                        if k.get("summary"):
                            print(f"    ↳ {DIM}{k['summary']}{RESET}")
                    print()
                else:
                    parts = [p.strip() for p in arg.split("|", 1)]
                    name = parts[0]
                    summary = parts[1] if len(parts) > 1 else ""
                    kp = await container.learning_store.upsert_knowledge_point(
                        course_id="default", name=name, summary=summary, user_id=user_id
                    )
                    print(f"{ts()} {GREEN}✅ 已保存知识点: [{kp['id']}] {kp['name']}{RESET}\n")
                continue

            if raw_input.startswith("/quiz"):
                arg = raw_input[len("/quiz") :].strip()
                if not arg:
                    print(f"{RED}用法: /quiz <知识点或主题> [choice|fill|short_answer]{RESET}")
                    continue
                parts = arg.split()
                q_type = "choice"
                if len(parts) > 1 and parts[-1] in ("choice", "fill", "short_answer"):
                    q_type = parts[-1]
                    topic = " ".join(parts[:-1])
                else:
                    topic = arg
                print(f"{ts()} {BLUE}正在围绕「{topic}」生成测验题 ({q_type})...{RESET}")
                quiz_tool = container.tool_registry.get("quiz_tool")
                ctx = ToolContext(user_id=user_id, session_id=session.session_id, rag_namespace=session.rag_namespace or "default", course_id=session.course_id)
                res = await quiz_tool.run({"action": "generate", "topic": topic, "question_type": q_type}, ctx)
                if not res.ok:
                    print(f"{RED}出题失败: {res.error}{RESET}")
                    continue
                print(f"\n{GREEN}{res.output}{RESET}")
                print(f"{DIM}提示：可直接输入你的答案，或使用 /answer <答案> 作答。{RESET}\n")
                continue

            if raw_input.startswith("/answer"):
                ans = raw_input[len("/answer") :].strip()
                if not ans:
                    print(f"{RED}用法: /answer <你的答案>{RESET}")
                    continue
                grade_tool = container.tool_registry.get("grade_tool")
                ctx = ToolContext(user_id=user_id, session_id=session.session_id, rag_namespace=session.rag_namespace or "default", course_id=session.course_id)
                res = await grade_tool.run({"user_answer": ans}, ctx)
                if not res.ok:
                    print(f"{RED}判卷失败: {res.error}{RESET}")
                    continue
                print(f"\n{CYAN}{res.output}{RESET}\n")
                continue

            if raw_input == "/review":
                review_tool = container.tool_registry.get("review_tool")
                ctx = ToolContext(user_id=user_id, session_id=session.session_id, rag_namespace=session.rag_namespace or "default", course_id=session.course_id)
                res = await review_tool.run({"action": "due"}, ctx)
                print(f"\n{CYAN}{res.output}{RESET}\n")
                continue

            if raw_input == "/progress":
                progress_tool = container.tool_registry.get("progress_tool")
                ctx = ToolContext(user_id=user_id, session_id=session.session_id, rag_namespace=session.rag_namespace or "default", course_id=session.course_id)
                res = await progress_tool.run({"action": "report"}, ctx)
                print(f"\n{CYAN}{res.output}{RESET}\n")
                continue

            if raw_input.startswith("/plan"):
                goal = raw_input[len("/plan") :].strip()
                progress_tool = container.tool_registry.get("progress_tool")
                ctx = ToolContext(user_id=user_id, session_id=session.session_id, rag_namespace=session.rag_namespace or "default", course_id=session.course_id)
                if not goal:
                    res = await progress_tool.run({"action": "list_plans"}, ctx)
                    print(f"\n{CYAN}{res.output}{RESET}\n")
                else:
                    print(f"{ts()} {BLUE}正在基于学情画像生成「{goal}」学习计划...{RESET}")
                    res = await progress_tool.run({"action": "create_plan", "goal": goal}, ctx)
                    if not res.ok:
                        print(f"{RED}生成计划失败: {res.error}{RESET}")
                        continue
                    print(f"\n{GREEN}{res.output}{RESET}\n")
                continue

            if raw_input.startswith("/ingest "):
                fpath_str = raw_input.split(" ", 1)[1].strip()
                fpath = Path(fpath_str)
                if not fpath.exists():
                    print(f"{RED}文件不存在: {fpath_str}{RESET}")
                    continue
                print(f"{ts()} {BLUE}正在导入文档: {fpath_str} ...{RESET}")
                doc_id, chunks = await container.ingestion_pipeline.ingest_file(
                    fpath, namespace=session.rag_namespace or "default"
                )
                print(f"{ts()} {GREEN}文档导入成功！DocID: {doc_id}, 切块数: {chunks}{RESET}")
                continue

            images_list: list[str] | None = None
            user_msg = raw_input
            if raw_input.startswith("/image"):
                import re as _re

                rest = raw_input[len("/image") :].strip()
                # 支持 /image "C:\path with space\a.jpg" 问题 或 /image C:\a.jpg 问题
                m = _re.match(r'^["\']([^"\']+)["\']\s*(.*)$', rest) or _re.match(r"^(\S+)\s*(.*)$", rest)
                if not m:
                    print(f"{RED}用法: /image <图片路径> [问题]{RESET}")
                    continue
                img_path = Path(m.group(1).strip())
                if not img_path.exists():
                    print(f"{RED}图片文件不存在: {img_path}{RESET}")
                    continue
                images_list = [str(img_path)]
                user_msg = m.group(2).strip() or "请详细描述这张图片的内容。"
            else:
                # 自动识别对话中带引号的本地图片路径，如 "C:\Users\...\xxx.jpg"
                import re as _re

                quoted_imgs = _re.findall(
                    r'["\']([a-zA-Z]:\\[^"\']+\.(?:jpg|jpeg|png|webp|bmp|gif))["\']',
                    raw_input,
                    flags=_re.IGNORECASE,
                )
                valid_imgs = [p for p in quoted_imgs if Path(p).exists()]
                if valid_imgs:
                    images_list = valid_imgs
                    print(f"{ts()} 🖼️  {CYAN}已自动检测到本地图片并启用原生多模态传入: {valid_imgs}{RESET}")

            # 发送聊天请求并实时打印事件
            print(f"\n{DIM}--- 开始推理步骤 ---{RESET}")
            req = ChatRequest(
                user_id=user_id,
                session_id=session.session_id,
                message=user_msg,
                images=images_list,
            )
            final_token_streamed = False

            async for ev in container.agent_loop.run(req, session):
                if ev.type == EventType.STATUS:
                    print(f"{ts()} ℹ️  {BLUE}[{ev.data.get('stage')}]{RESET} {ev.data.get('detail')}")
                elif ev.type == EventType.THOUGHT:
                    print(f"{ts()} 💭 {MAGENTA}[Thought - 步 {ev.step}]{RESET} {ev.data.get('text')}")
                elif ev.type == EventType.TOOL_CALL:
                    args_fmt = json.dumps(ev.data.get("arguments", {}), ensure_ascii=False)
                    print(f"{ts()} 🛠️  {YELLOW}[Action - 调用工具]{RESET} {ev.data.get('name')}({args_fmt})")
                elif ev.type == EventType.OBSERVATION:
                    ok = ev.data.get("ok", True)
                    st_symbol = f"{GREEN}成功{RESET}" if ok else f"{RED}失败{RESET}"
                    out_snippet = ev.data.get("output", "")
                    if len(out_snippet) > 200:
                        out_snippet = out_snippet[:200] + "..."
                    print(f"{ts()} 👁️  {CYAN}[Observation - {st_symbol}]{RESET} {out_snippet}")
                elif ev.type == EventType.REFLECTION:
                    done_text = f"{GREEN}已充分{RESET}" if ev.data.get("done") else f"{YELLOW}需深入{RESET}"
                    print(f"{ts()} 🪞 [Reflection - 反思评估] 状态: {done_text} | 评价: {ev.data.get('critique')}")
                elif ev.type == EventType.TOKEN:
                    if not final_token_streamed:
                        print(f"\n{ts()} 🤖 {GREEN}{BOLD}[Final Answer]{RESET}\n", end="", flush=True)
                        final_token_streamed = True
                    print(ev.data.get("text", ""), end="", flush=True)
                elif ev.type == EventType.FINAL:
                    if not final_token_streamed:
                        print(f"\n{ts()} 🤖 {GREEN}{BOLD}[Final Answer]{RESET}\n{ev.data.get('answer')}")
                    else:
                        print()
                elif ev.type == EventType.DONE:
                    print(f"\n{ts()} ⏱️  {DIM}[Done - 耗时 {ev.data.get('elapsed_sec')}s]{RESET}\n")

    finally:
        await container.aclose()


def main():
    parser = argparse.ArgumentParser(description="智能体助手简易终端交互 (CLI)")
    parser.add_argument("--user", default="user_demo", help="用户 ID (默认: user_demo)")
    args = parser.parse_args()

    print_banner()
    asyncio.run(run_local_cli(user_id=args.user))


if __name__ == "__main__":
    main()
