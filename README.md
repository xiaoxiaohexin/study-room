# 

本项目基于开源项目 Hello Agent 文档学习后搭建后端，前端采用 llama.cpp 的 UI 界面修改。

搭建本项目的初衷是为了方便我妹学习，还有就是感觉 Google 的 notebook 用的很别扭，虽然响应快并且逻辑清晰，但是对于回顾之前的内容很不方便，然后就自己动手搭建一个本地的、真正适合平时学习节奏的项目。

---

##  为什么做这个项目

开学的时候我接触了Hello Agent 这个项目，首先我就是用Google的notebook进行学习，不可否认NotebookLM 确实有很多亮点（总结快、逻辑清楚、反应敏捷），但用着用着总觉得有几个很麻烦的点

1.作为一个学习应用，上下文不能切断回顾，看记录只能滑动查找，特别浪费时间，而且查起来特容易断思绪如果深入一个话题后，再想回到之前的话题，会很麻烦。
2.不能利用本地的资源，需要联网，作为一个计算机学生，学习怎么能不兼容本地，而且上传很麻烦，不能点击加号快速上传验证，需要点击添加来源，验证完后要及时删除，否则会影响LLM的判断
3.我习惯用思维导图等逻辑理解知识点，但是我发现我对NotebookLM里面的Studio功能无法描述自己的需求，有点力不从心。

正好现在正在学习开源项目 **Hello Agent** ，了解的Agent的运行逻辑，就打算练练手，感觉llama.cpp的桌面非常符合我的审美就拿来直接套用了，感谢llama.cpp的开源！！！

## 📁 目录结构

```text
xiaoxiao/
├── app/
│   ├── agent/               # Agent 核心逻辑：ReAct 循环 (loop.py)、系统 Prompt 与反思
│   ├── api/                 # API 路由：兼容 llama.cpp (/props, completions, /mcp) 及学习专用路由
│   ├── context/             # GSSC 上下文构建器（KV Cache 友好的提示词组装与 Token 控制）
│   ├── core/                # 核心配置 (config.py)、依赖注入容器 (container.py)、生命周期
│   ├── learning/            # 学习算法核心：SM-2 间隔重复、EMA 掌握度、IntentRouter、学习存储库
│   ├── llm/                 # LLM 客户端抽象、多模态解析、OpenAI 协议封装与结构化输出
│   ├── memory/              # 四层记忆体系：工作记忆、感知记忆、语义记忆、情境记忆及生命周期
│   ├── observability/       # 日志追踪与对话日志记录
│   ├── query/               # 查询扩展（MQE 多查询生成与 HyDE 假设性文档检索）
│   ├── rag/                 # 教材文档切片 (chunker)、加载 (loader)、管道与混合检索 (retriever)
│   ├── schemas/             # Pydantic 数据规范与 SSE 事件协议
│   ├── session/             # 会话状态管理器
│   ├── storage/             # 存储适配器：SQLite（含自研余弦向量检索）、Qdrant、Neo4j、Redis 缓存
│   ├── tools/               # 工具系统：学习专属工具（出题/判题/复习/看板）、RAG 工具、终端/笔记工具
│   └── main.py              # FastAPI 应用入口与中间件挂载
├── llama-webui/             # 基于 llama.cpp 官方 WebUI 定制的前端工程 (SvelteKit + TailwindCSS)
│   └── tools/ui/            # WebUI 源代码
├── data/                    # 本地数据目录：agent.db、教材上传目录、错题与笔记记录
├── tests/                   # 单元测试与端到端集成测试集
├── cli.py                   # 伴学智能体彩色交互式终端（单文件开箱即用体验）
├── pyproject.toml           # 后端 Python 项目依赖与配置
├── docker-compose.yml       # 可选：快速启动 Qdrant/Neo4j/Redis 服务的 Compose 文件
└── .env.example             # 环境变量配置示例
```

---

## 🚀 快速上手与运行指南

系统支持完全在本地离线单机运行。

### 第一步：环境准备
- **Python**: 推荐 3.11 或 3.12
- **Node.js**: 推荐 18.x 或以上（用于运行 WebUI）
- **包管理工具**: 推荐 `uv`（极速）或者直接使用 `pip`

```bash
# 1. 克隆或进入项目根目录
cd e:\xiaoxiao

# 2. 安装 Python 后端依赖
pip install -e .
# 或者使用 uv：
# uv sync
```

### 第二步：配置环境与模型
复制环境变量配置文件：
```bash
cp .env.example .env
```
打开 `.env` 文件，根据实际情况配置主模型与 Embedding 模型地址。
- **本地部署推荐**：
  - 启动主推理模型（如 Gemma系列或 Qwen系列）：
    `llama-server -m your-model.gguf --port 8083` 当然啦，也可以用Ollama，只要是OpenAI协议的就行
  - 启动向量模型（如 BGE 或 Qwen-Embedding）：
    `llama-server -m your-embed-model.gguf --port 8084 --embedding` 嗯，用于RAG检索的
- **云端 API 兼容**：
  如需使用 DeepSeek 或其他云端大模型，直接修改 `.env` 中的 `LLM_BASE_URL` 与 `LLM_API_KEY` 即可。推荐多模态模型哦
- **外部存储**：
  若不配置 Qdrant / Neo4j / Redis，系统将自动无缝降级使用本地 SQLite，免去配置容器的烦恼！

---

### 第三步：运行方式

你可以根据使用习惯，自由选择通过 **WebUI 界面** 或者 **终端 CLI** 来学习：

#### 方式 A：启动后端服务 + WebUI 界面（日常推荐）

1. **启动 FastAPI 后端**（负责路由代理、RAG 检索、记忆管理与 MCP 服务）：
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

2. **启动前端 WebUI**：
```bash
cd llama-webui/tools/ui
npm install
npm run dev
```
打开浏览器访问 `http://localhost:5173`。在设置中确认连接到 `http://127.0.0.1:8000`，即可享受沉浸式学习体验！

---

#### 方式 B：直接使用彩色交互终端 CLI （当时没有前端，随便搞出来验证的）

无需打开浏览器，在命令行就能用。终端实时思考阶段展示和快捷指令支持：

```bash
python cli.py --user xiaoxiao
```
启动后会打印欢迎横幅与组件健康自检，即可进入对话。

---

## ⌨️ 终端 CLI 快捷指令速查表

在 `cli.py` 终端中，除了可以直接像老师一样聊天提问，还预置了丰富的学习控制指令：

| 指令 | 示例 | 功能说明 |
| :--- | :--- | :--- |
| **自由对话** | `光合作用的暗反应过程是什么？` | 自动识别意图：查阅讲义并展开详细精讲 |
| `/quiz` | `/quiz 三角函数诱导公式 choice` | 针对指定知识点生成练习题（可选 `choice` 单选 / `fill` 填空 / `short_answer` 简答） |
| `/answer` | `/answer B` 或 `/answer 线粒体` | 提交题目答案，自动触发判卷、计算掌握度与安排复习时间 |
| `/review` | `/review` | 查看今天到期需要复习的知识点与遗留错题清单 |
| `/progress`| `/progress` | 查看当前个人学情报告、掌握度分布与最薄弱知识点 |
| `/plan` | `/plan 期末函数冲刺` | 基于当前薄弱点智能规划阶段性复习方案 |
| `/kp` | `/kp` 或 `/kp 欧姆定律 \| 核心公式与电阻串并联` | 查看已有知识点图谱或录入新知识点 |
| `/course` | `/course` 或 `/course 高中生物` | 查看课程列表或切换/新建学习课程 |
| `/image` | `/image "D:\homework\p1.jpg" 这题该怎么列辅助线？` | 多模态拍照搜题辅导（苏格拉底式启发引导） |
| `/ingest` | `/ingest "D:\books\必修一讲义.pdf"` | 将本地教材、课件或笔记导入 RAG 向量库 |
| `/memory` | `/memory` | 查看当前智能体为你沉淀的四层记忆 |
| `/tools` | `/tools` | 查看系统当前已挂载激活的所有学习与系统工具 |
| `/new` | `/new` | 开启一个全新的对话会话 |
| `/exit` | `/exit` 或 `/quit` | 退出交互终端 |



## 致谢与参考

- 核心架构深度参考了开源项目 **[Hello Agent](https://github.com/datawhalechina/hello-agent)**，感谢教程团队把 Agent 的 ReAct 循环、多层记忆与工具调度讲解得通俗透彻！
- 前端与推理底座基于卓越的开源项目 **[llama.cpp](https://github.com/ggerganov/llama.cpp)**，感谢开源社区带来的高效本地推理与优秀的 WebUI。
