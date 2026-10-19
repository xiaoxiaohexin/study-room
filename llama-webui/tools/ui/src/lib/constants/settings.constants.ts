import { DEFAULT_AGENTIC_CONFIG } from './agentic.constants';
import { CLI_FLAGS } from './cli-flags.constants';
import { DEFAULT_MCP_CONFIG } from './mcp.constants';
import { SETTINGS_KEYS } from './settings-keys.constants';
import { TITLE_GENERATION } from './title-generation.constants';
import { FILE_GLOB_SEARCH_PICKERS } from './working-directory.constants';
import {
	Code,
	Database,
	Funnel,
	ListRestart,
	Monitor,
	Moon,
	PencilRuler,
	SlidersVertical,
	Sun
} from '@lucide/svelte';
import { SyncableParameterType } from '$lib/enums';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { ColorMode } from '$lib/enums/ui.enums';
import type {
	SettingsConfigValue,
	SettingsEntry,
	SettingsFieldConfig,
	SettingsSection,
	SettingsSectionEntry
} from '$lib/types';

/** Settings sections — slug is the routing identity, title is the display label. */
export const SETTINGS_SECTIONS = {
	AGENTIC: { slug: 'agentic', title: '智能体（Agentic）' },
	DEVELOPER: { slug: 'developer', title: '开发者选项' },
	DISPLAY: { slug: 'display', title: '界面显示' },
	GENERAL: { slug: 'general', title: '常规设置' },
	IMPORT_EXPORT: { slug: 'import-export', title: '导入 / 导出' },
	SAMPLING_PENALTIES: { slug: 'sampling-penalties', title: '采样与惩罚参数' },
	TOOLS: { slug: 'tools', title: '工具（Tools）' }
} as const;

export const SETTINGS_SECTION_SLUGS = {
	AGENTIC: SETTINGS_SECTIONS.AGENTIC.slug,
	DEVELOPER: SETTINGS_SECTIONS.DEVELOPER.slug,
	DISPLAY: SETTINGS_SECTIONS.DISPLAY.slug,
	GENERAL: SETTINGS_SECTIONS.GENERAL.slug,
	IMPORT_EXPORT: SETTINGS_SECTIONS.IMPORT_EXPORT.slug,
	SAMPLING_PENALTIES: SETTINGS_SECTIONS.SAMPLING_PENALTIES.slug,
	TOOLS: SETTINGS_SECTIONS.TOOLS.slug
} as const;

export const SETTINGS_SECTION_TITLES = {
	AGENTIC: SETTINGS_SECTIONS.AGENTIC.title,
	DEVELOPER: SETTINGS_SECTIONS.DEVELOPER.title,
	DISPLAY: SETTINGS_SECTIONS.DISPLAY.title,
	GENERAL: SETTINGS_SECTIONS.GENERAL.title,
	IMPORT_EXPORT: SETTINGS_SECTIONS.IMPORT_EXPORT.title,
	SAMPLING_PENALTIES: SETTINGS_SECTIONS.SAMPLING_PENALTIES.title,
	TOOLS: SETTINGS_SECTIONS.TOOLS.title
} as const;

export const SETTINGS_REGISTRY: SettingsSectionEntry[] = [
	// General
	{
		icon: SlidersVertical,
		settings: [
			{
				defaultValue: ColorMode.SYSTEM,
				help: '选择界面颜色主题。可选择跟随系统（System）、浅色（Light）或深色（Dark）。',
				key: SETTINGS_KEYS.THEME,
				label: '界面主题',
				options: [
					{ icon: Monitor, label: '跟随系统', value: ColorMode.SYSTEM },
					{ icon: Sun, label: '浅色模式', value: ColorMode.LIGHT },
					{ icon: Moon, label: '深色模式', value: ColorMode.DARK }
				],
				type: SettingsFieldType.SELECT
			},
			{
				defaultValue: '',
				help: `如果服务端启动时配置了 <code> ${CLI_FLAGS.API_KEY} </code> 选项，请在此填写对应的 API 密钥。`,
				isPrivate: true,
				key: SETTINGS_KEYS.API_KEY,
				label: 'API 密钥（API Key）',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: '',
				help: '定义模型角色与行为准则的初始系统提示词（System Prompt）。',
				key: SETTINGS_KEYS.SYSTEM_MESSAGE,
				label: '系统提示词（System Message）',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: true,
				help: '在每个会话顶部显示系统提示词区域。',
				key: SETTINGS_KEYS.SHOW_SYSTEM_MESSAGE,
				label: '显示系统提示词',
				standaloneField: false,
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: 2500,
				help: '粘贴超长文本时自动转换为文本附件。可通过此参数设置触发转换的字符长度阈值（设为 0 则禁用）。',
				key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
				label: '长文本粘贴转附件长度阈值',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: true,
				help: '按 Enter 键发送消息，按 Shift + Enter 换行。关闭后需使用 Ctrl/Cmd + Enter 发送。',
				key: SETTINGS_KEYS.SEND_ON_ENTER,
				label: '按 Enter 键发送消息',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '当模型支持音频输入且输入框为空时，自动将发送按钮切换为麦克风语音输入按钮。',
				key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
				label: '输入框为空时显示麦克风按钮',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '为助手回复（含深度思考模型）启用“继续生成（Continue）”按钮。',
				isExperimental: true,
				key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
				label: '启用“继续生成”按钮',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在侧边栏显示“发现模型”入口以浏览和下载 HuggingFace GGUF 模型（仅在路由模式下可用）。',
				key: SETTINGS_KEYS.ENABLE_DISCOVER_MODELS,
				label: '启用发现模型（Discover Models）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '从 Hugging Face Hub API 获取模型展示元数据（图标、上下文长度、对话模板等）。',
				key: SETTINGS_KEYS.USE_HUGGING_FACE_HUB,
				label: '使用 Hugging Face Hub API',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '选择会话标题的生成方式：首行非空文本使用快速确定性规则；LLM 选项则根据首轮对话由模型自动总结标题。',
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
				label: '会话标题生成方式',
				radioOptions: [
					{
						key: SETTINGS_KEYS.TITLE_GENERATION_USE_FIRST_LINE,
						label: '使用首行非空文本作为会话标题',
						value: 'firstLine'
					},
					{
						isExperimental: true,
						key: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
						label: '使用 LLM 自动生成会话标题',
						value: 'llm'
					}
				],
				type: SettingsFieldType.RADIO
			},
			{
				defaultValue: TITLE_GENERATION.DEFAULT_PROMPT,
				dependsOn: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
				help: '可选的标题生成提示词模板。使用 {{USER}} 代表用户消息，{{ASSISTANT}} 代表助手回复。',
				key: SETTINGS_KEYS.TITLE_GENERATION_PROMPT,
				label: 'LLM 标题生成提示词模板',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: false,
				help: '会话标题生成单选框的同步存储项。',
				key: SETTINGS_KEYS.TITLE_GENERATION_USE_LLM,
				label: '使用 LLM 自动生成标题',
				standaloneField: false,
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '复制带有文本附件的消息时，将其合并为纯文本字符串，而不是可粘贴回附件的特殊格式。',
				key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
				label: '将文本附件复制为纯文本',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '将 PDF 解析为图片而非提取文本（非视觉模型会自动回退为文本解析）。',
				key: SETTINGS_KEYS.PDF_AS_IMAGE,
				label: '将 PDF 作为图片解析',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: 0,
				help: '超过此像素分辨率的图片在发送给服务端前会自动缩放。设为 0 表示不限制。',
				key: SETTINGS_KEYS.MAX_IMAGE_RESOLUTION,
				label: '最大图片分辨率（百万像素 MP）',
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.GENERAL,
		title: SETTINGS_SECTION_TITLES.GENERAL
	},
	// Display
	{
		icon: Monitor,
		settings: [
			{
				defaultValue: true,
				help: '在每条助手回复下方显示生成速率统计（Token/秒、Token 总数、耗时）。',
				key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				label: '显示消息生成统计信息',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				dependsOn: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
				help: '在多步工具调用（Agentic）响应中，显示每一轮子步骤的耗时与 Token 统计。',
				key: SETTINGS_KEYS.SHOW_AGENTIC_TURN_STATS,
				label: '显示单步 Agent 轮次统计信息',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '生成消息时默认展开模型的思考过程（Thought Process）。',
				key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
				label: '生成时自动展开思考过程',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '执行工具调用时自动展开详情，并在执行完成后保持展开状态。',
				key: SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT,
				label: '始终展开工具调用详情',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在聊天记录中以 Markdown 格式渲染用户发送的消息。',
				key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
				label: '将用户消息渲染为 Markdown',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '将深度思考（Reasoning/Thinking）块中的内容渲染为 Markdown 格式而非纯文本。',
				key: SETTINGS_KEYS.RENDER_THINKING_AS_MARKDOWN,
				label: '将思考过程渲染为 Markdown',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '始终按完整高度展示代码块，取消最大高度滚动限制。',
				key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
				label: '代码块完整高度展示',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '流式输出消息时禁用自动滚动到底部，方便手动浏览上文。',
				key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
				label: '禁用流式自动滚动',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在桌面端始终保持左侧导航栏展开，不自动隐藏。',
				key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
				label: '桌面端常驻显示侧边栏',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在会话顶部显示类似浏览器的多标签页栏，方便在多个已打开的对话间快速切换。',
				key: SETTINGS_KEYS.CONVERSATION_TABS,
				label: '启用多会话标签页（Tabs）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '显示完整的原始模型标识符（如 "ggml-org/GLM-4.7-Flash-GGUF:Q8_0"），而不是解析后的简短名称与徽章。',
				key: SETTINGS_KEYS.SHOW_RAW_MODEL_NAMES,
				label: '显示原始模型全称',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在界面模型名称旁显示量化精度徽章（如 Q8_0、Q4_K_M）。',
				key: SETTINGS_KEYS.SHOW_MODEL_QUANTIZATION,
				label: '显示模型量化精度标签',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在界面模型名称旁显示模型特性标签（如 "vision"、"reasoning"）。',
				key: SETTINGS_KEYS.SHOW_MODEL_TAGS,
				label: '显示模型特性标签',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在模型选择下拉列表行中显示模态与能力图标（视觉、音频、思考、工具）。',
				key: SETTINGS_KEYS.SHOW_MODEL_CAPABILITIES_IN_SELECTOR,
				label: '在模型选择器中显示模态与能力图标',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在模型选择器中并列显示模型的发布组织/作者名称。',
				key: SETTINGS_KEYS.SHOW_MODEL_ORG_NAME,
				label: '在模型选择器中显示组织名称',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在界面右下角显示当前构建版本号信息。',
				key: SETTINGS_KEYS.SHOW_BUILD_VERSION,
				label: '显示构建版本信息',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在文件与文件夹 @ 引用徽章中显示完整文件系统路径，而非仅显示文件名。',
				key: SETTINGS_KEYS.SHOW_FULL_PATH_IN_MENTIONS,
				label: '在 @ 引用中显示完整路径',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: true,
				help: '在模型选择器与模型管理器中按模型家族（如 "Qwen"、"Gemma"）分组展示。',
				key: SETTINGS_KEYS.GROUP_MODELS_BY_FAMILY,
				label: '按模型家族分组展示',
				type: SettingsFieldType.CHECKBOX
			}
		],
		slug: SETTINGS_SECTION_SLUGS.DISPLAY,
		title: SETTINGS_SECTION_TITLES.DISPLAY
	},
	// MCP Servers (non-UI config object)
	{
		icon: PencilRuler,
		settings: [
			{
				defaultValue: '[]',
				help: '以 JSON 列表配置 MCP 服务端。请在 MCP 服务面板中进行可视化编辑。',
				key: SETTINGS_KEYS.MCP_SERVERS,
				label: 'MCP 服务端列表',
				standaloneField: false,
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.TOOLS,
		title: SETTINGS_SECTION_TITLES.TOOLS
	},
	// Tools
	{
		icon: ListRestart,
		settings: [
			{
				defaultValue: DEFAULT_AGENTIC_CONFIG.maxTurns,
				help: '单次回复中允许的最大工具循环调用轮数（防止无限循环）。',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.AGENTIC_MAX_TURNS,
				label: '最大智能体工具调用轮数（Agentic Turns）',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: DEFAULT_MCP_CONFIG.requestTimeoutSeconds,
				help: '单次 MCP 工具调用的超时时间（秒）。',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.MCP_REQUEST_TIMEOUT_SECONDS,
				label: 'MCP 请求超时时间（秒）',
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: FILE_GLOB_SEARCH_PICKERS.DEFAULT_SEARCH_DEPTH,
				help: '@ 引用文件搜索在工作目录下递归查找的最大目录层级深度。',
				isPositiveInteger: true,
				key: SETTINGS_KEYS.MENTION_SEARCH_MAX_DEPTH,
				label: '@ 引用文件搜索深度',
				max: FILE_GLOB_SEARCH_PICKERS.MAX_SEARCH_DEPTH,
				min: 1,
				placeholder: `${FILE_GLOB_SEARCH_PICKERS.DEFAULT_SEARCH_DEPTH}`,
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.AGENTIC,
		title: SETTINGS_SECTION_TITLES.AGENTIC
	},
	// Import/Export
	{
		icon: Database,
		settings: [],
		slug: SETTINGS_SECTION_SLUGS.IMPORT_EXPORT,
		title: SETTINGS_SECTION_TITLES.IMPORT_EXPORT
	},
	// Sampling
	{
		icon: Funnel,
		settings: [
			{
				defaultValue: undefined,
				help: '控制生成文本的随机性（影响输出 Token 的概率分布）。值越高越随机、越具创造性；值越低越确定、越聚焦。',
				key: SETTINGS_KEYS.TEMPERATURE,
				label: '温度（Temperature）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.TEMPERATURE
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '温度采样器的扩展项：动态温度范围附加值，根据 Token 熵动态调整概率。',
				key: SETTINGS_KEYS.DYNATEMP_RANGE,
				label: '动态温度范围（Dynatemp Range）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DYNATEMP_RANGE
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '温度采样器的扩展项：基于最高概率 Token 平滑概率重分布的指数。',
				key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
				label: '动态温度指数（Dynatemp Exponent）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DYNATEMP_EXPONENT
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '仅保留概率排名前 K 的候选 Token。',
				key: SETTINGS_KEYS.TOP_K,
				label: 'Top K',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TOP_K },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '核采样（Nucleus Sampling）：仅保留累积概率达到 P 的最小候选 Token 集合。',
				key: SETTINGS_KEYS.TOP_P,
				label: 'Top P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TOP_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '相对于最高概率 Token 的最小概率比例阈值，低于该比例的 Token 将被过滤。',
				key: SETTINGS_KEYS.MIN_P,
				label: 'Min P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.MIN_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'XTC 采样器会剔除顶部高概率 Token 以提升多样性；此参数控制触发剔除的概率（0 表示禁用 XTC）。',
				key: SETTINGS_KEYS.XTC_PROBABILITY,
				label: 'XTC 概率（XTC Probability）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.XTC_PROBABILITY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'XTC 采样器剔除顶部 Token 所需的概率阈值。',
				key: SETTINGS_KEYS.XTC_THRESHOLD,
				label: 'XTC 阈值（XTC Threshold）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.XTC_THRESHOLD
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '局部典型采样（Locally Typical Sampling）：基于对数概率与信息熵的差异筛选 Token。',
				key: SETTINGS_KEYS.TYP_P,
				label: 'Typical P',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.TYP_P },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '单次回复生成的最大 Token 数量。设为 -1 表示不限制（直到上下文上限）。',
				key: SETTINGS_KEYS.MAX_TOKENS,
				label: '最大生成长度（Max Tokens）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.MAX_TOKENS
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: '',
				help: '采样器执行顺序，默认值为 "top_k;typ_p;top_p;min_p;temperature"。',
				key: SETTINGS_KEYS.SAMPLERS,
				label: '采样器执行链（Samplers）',
				sync: { paramType: SyncableParameterType.STRING, serverKey: SETTINGS_KEYS.SAMPLERS },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: false,
				help: '启用硬件加速后端采样。开启后，受支持的采样器将在 GPU/加速器后端直接执行以提升采样速度。',
				key: SETTINGS_KEYS.BACKEND_SAMPLING,
				label: '硬件后端采样（Backend Sampling）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: undefined,
				help: '计算重复惩罚时向前追溯的最近 Token 数量。',
				key: SETTINGS_KEYS.REPEAT_LAST_N,
				label: '重复惩罚窗口（Repeat Last N）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.REPEAT_LAST_N
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '控制生成文本中 Token 序列重复的惩罚力度（1.0 表示不惩罚）。',
				key: SETTINGS_KEYS.REPEAT_PENALTY,
				label: '重复惩罚系数（Repeat Penalty）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.REPEAT_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '存在惩罚：根据 Token 是否已在输出中出现过施加固定惩罚。',
				key: SETTINGS_KEYS.PRESENCE_PENALTY,
				label: '存在惩罚（Presence Penalty）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.PRESENCE_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '频率惩罚：根据 Token 在输出中出现的频次施加递增惩罚。',
				key: SETTINGS_KEYS.FREQUENCY_PENALTY,
				label: '频率惩罚（Frequency Penalty）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.FREQUENCY_PENALTY
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY（Don’t Repeat Yourself）采样可在长上下文中抑制重复生成。此参数设置 DRY 惩罚乘数。',
				key: SETTINGS_KEYS.DRY_MULTIPLIER,
				label: 'DRY 惩罚乘数（DRY Multiplier）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_MULTIPLIER
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY 采样指数底数（Base Value）。',
				key: SETTINGS_KEYS.DRY_BASE,
				label: 'DRY 底数（DRY Base）',
				sync: { paramType: SyncableParameterType.NUMBER, serverKey: SETTINGS_KEYS.DRY_BASE },
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: '触发 DRY 指数重复惩罚前允许的最大重复序列长度。',
				key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
				label: 'DRY 允许重复长度（DRY Allowed Length）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_ALLOWED_LENGTH
				},
				type: SettingsFieldType.INPUT
			},
			{
				defaultValue: undefined,
				help: 'DRY 采样扫描重复序列的最近 N 个 Token 范围。',
				key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
				label: 'DRY 惩罚窗口（DRY Penalty Last N）',
				sync: {
					paramType: SyncableParameterType.NUMBER,
					serverKey: SETTINGS_KEYS.DRY_PENALTY_LAST_N
				},
				type: SettingsFieldType.INPUT
			}
		],
		slug: SETTINGS_SECTION_SLUGS.SAMPLING_PENALTIES,
		title: SETTINGS_SECTION_TITLES.SAMPLING_PENALTIES
	},
	// Developer
	{
		icon: Code,
		settings: [
			{
				defaultValue: false,
				help: '每次回复结束后，自动将当前会话重新提交给服务端预填 KV Cache，从而加快下一轮提问的首字响应速度。',
				key: SETTINGS_KEYS.PRE_ENCODE_CONVERSATION,
				label: '回复后预填充 KV 缓存（Pre-fill KV Cache）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '发送 reasoning_format=none，使服务端直接内联返回思考 Token，而不将其提取到独立的思考字段。',
				key: SETTINGS_KEYS.DISABLE_REASONING_PARSING,
				label: '禁用思考内容解析（Disable Reasoning Parsing）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在发送新请求前剔除历史消息中的思考过程（Thinking）。关闭时会将思考过程通过 reasoning_content 字段传回模型。',
				key: SETTINGS_KEYS.EXCLUDE_REASONING_FROM_CONTEXT,
				label: '从上下文中排除历史思考内容',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '在每条消息旁显示切换按钮，支持将消息切换为未格式化的原始纯文本视图。',
				key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
				label: '启用原始输出切换按钮（Raw Output）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				help: '向模型暴露 run_javascript 浏览器沙箱工具。代码运行在隔离的 Web Worker 沙箱 iframe 中，具有严格超时保护。',
				key: SETTINGS_KEYS.JS_SANDBOX_ENABLED,
				label: '启用 JavaScript 沙箱工具',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: false,
				dependsOn: SETTINGS_KEYS.JS_SANDBOX_ENABLED,
				help: '在沙箱中预加载 nerdamer 符号数学计算库（支持化简、求导、积分、解方程等）。需要先启用“JavaScript 沙箱工具”。',
				key: SETTINGS_KEYS.SYMBOLIC_MATH_ENABLED,
				label: '启用符号数学库（nerdamer）',
				type: SettingsFieldType.CHECKBOX
			},
			{
				defaultValue: '',
				help: '附加发送给 API 的自定义 JSON 请求参数（必须是合法的 JSON 格式）。',
				key: SETTINGS_KEYS.CUSTOM_JSON,
				label: '自定义请求参数（Custom JSON）',
				type: SettingsFieldType.TEXTAREA
			},
			{
				defaultValue: '',
				help: '运行时注入页面的自定义 CSS 样式。',
				key: SETTINGS_KEYS.CUSTOM_CSS,
				label: '自定义样式（Custom CSS）',
				type: SettingsFieldType.TEXTAREA
			}
		],
		slug: SETTINGS_SECTION_SLUGS.DEVELOPER,
		title: SETTINGS_SECTION_TITLES.DEVELOPER
	}
];

function getAllSettings(): SettingsEntry[] {
	const result: SettingsEntry[] = [];

	for (const section of SETTINGS_REGISTRY) {
		result.push(...section.settings);
	}

	return result;
}

/** Flat config object stored in localStorage. */
export const SETTING_CONFIG_DEFAULT: Record<string, SettingsConfigValue> = Object.fromEntries(
	getAllSettings().map((s) => [s.key, s.defaultValue])
) as Record<string, SettingsConfigValue>;

/** Help text for every setting (including non-UI). */
export const SETTING_CONFIG_INFO: Record<string, string> = Object.fromEntries(
	getAllSettings().map((s) => [s.key, s.help])
) as Record<string, string>;

/** Sidebar sections + field configs (as consumed by UI). */
function toSettingsSection(section: SettingsSectionEntry): SettingsSection {
	return {
		fields: section.settings
			.filter((s) => s.standaloneField !== false)
			.map((s) => ({
				dependsOn: s.dependsOn,
				help: s.help,
				isExperimental: s.isExperimental,
				isPositiveInteger: s.isPositiveInteger,
				isPrivate: s.isPrivate,
				key: s.key,
				label: s.label,
				max: s.max,
				min: s.min,
				options: s.options as SettingsFieldConfig['options'],
				placeholder: s.placeholder,
				radioOptions: s.radioOptions,
				type: s.type
			})),
		icon: section.icon,
		slug: section.slug,
		title: section.title
	};
}

/** Sidebar sections in custom display order (the registry array order). */
export const SETTINGS_CHAT_SECTIONS: SettingsSection[] = SETTINGS_REGISTRY.map(toSettingsSection);

/** INPUT-type settings whose value is a number. */
export const NUMERIC_FIELDS = getAllSettings()
	.filter((s) => s.type === SettingsFieldType.INPUT && typeof s.defaultValue !== 'string')
	.map((s) => s.key) as readonly string[];

/** Numeric fields clamped to >= 1 and rounded. */
export const POSITIVE_INTEGER_FIELDS = getAllSettings()
	.filter((s) => s.isPositiveInteger)
	.map((s) => s.key) as readonly string[];
