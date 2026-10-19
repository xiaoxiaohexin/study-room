<script lang="ts">
	import {
		BookOpen,
		Brain,
		ChevronDown,
		ChevronLeft,
		ChevronRight,
		Copy,
		FileText,
		Image as ImageIcon,
		LayoutGrid,
		Lightbulb,
		Link2,
		Loader2,
		Maximize2,
		MessageSquare,
		Plus,
		RefreshCw,
		RotateCcw,
		Save,
		Search,
		Sparkles,
		StickyNote,
		Trash2,
		X,
		ZoomIn,
		ZoomOut
	} from '@lucide/svelte';
	import { browser } from '$app/environment';
	import { base } from '$app/paths';
	import { MarkdownContent } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { AttachmentType, MessageRole } from '$lib/enums';
	import { chatStore, conversationsStore } from '$lib/stores';
	import type { DatabaseMessage, DatabaseMessageExtra } from '$lib/types';
	import { copyToClipboard } from '$lib/utils';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	export type CanvasViewMode = 'canvas' | 'split' | 'list';

	type NodeCardColor = 'amber' | 'sky' | 'emerald' | 'violet' | 'rose' | 'slate';
	type CustomNodeKind = 'note' | 'rag' | 'flashcard' | 'hub' | 'image';

	interface TurnImageItem {
		id: string;
		name: string;
		url: string;
		source: 'message' | 'custom' | 'linked' | 'inherited';
	}

	interface TurnDocItem {
		id: string;
		name: string;
		content: string;
		kind: string;
	}

	interface TurnMemoryItem {
		id: string;
		front: string;
		back: string;
		interval: number;
		easiness: number;
		flipped?: boolean;
	}

	interface TurnAssociationData {
		customImages?: Array<{ id: string; name: string; url: string }>;
		activeImageIndex?: number;
		inheritedImages?: boolean;
		imagePlacement?: 'top' | 'side' | 'hidden';
		ragItems?: Array<{ content: string; source: string; score: number }>;
		ragLoading?: boolean;
		ragExpanded?: boolean;
		memoryItems?: TurnMemoryItem[];
		memoryExpanded?: boolean;
		expandedDocId?: string | null;
	}

	interface CustomCanvasNode {
		id: string;
		kind: CustomNodeKind;
		title: string;
		content: string;
		backContent?: string;
		flipped?: boolean;
		ragQuery?: string;
		ragResults?: Array<{ content: string; source: string; score: number }>;
		ragLoading?: boolean;
		cardId?: string;
		sm2Interval?: number;
		sm2Easiness?: number;
		images?: Array<{ id: string; name: string; url: string }>;
		activeImageIndex?: number;
		linkedTurnId?: string;
		color: NodeCardColor;
		x: number;
		y: number;
		width: number;
	}

	interface CanvasEdge {
		id: string;
		from: string;
		to: string;
		label?: string;
		isAuto?: boolean;
	}

	interface PersistedCanvasData {
		panX: number;
		panY: number;
		zoom: number;
		nodePositions: Record<string, { x: number; y: number }>;
		customNodes: CustomCanvasNode[];
		customEdges: CanvasEdge[];
		turnAssociations?: Record<string, TurnAssociationData>;
	}

	interface Props {
		messages?: DatabaseMessage[];
		viewMode?: CanvasViewMode;
		onViewModeChange?: (mode: CanvasViewMode) => void;
		onInjectPrompt?: (text: string) => void;
	}

	let {
		messages = [],
		onInjectPrompt,
		onViewModeChange,
		viewMode = $bindable('canvas')
	}: Props = $props();

	const STORAGE_PREFIX = 'llama-infinite-canvas-v1:';

	let viewportEl: HTMLDivElement | undefined = $state();
	let imageFileInputEl: HTMLInputElement | undefined = $state();
	let uploadTarget = $state<{ type: 'turn' | 'node'; id: string } | null>(null);

	let panX = $state(80);
	let panY = $state(60);
	let zoom = $state(0.95);

	let isPanning = $state(false);
	let panStartMouse = $state({ x: 0, y: 0 });
	let panStartOffset = $state({ x: 0, y: 0 });

	let draggingNodeId = $state<string | null>(null);
	let dragNodeStartMouse = $state({ x: 0, y: 0 });
	let dragNodeStartPos = $state({ x: 0, y: 0 });

	let connectingFromId = $state<string | null>(null);
	let connectingMouseWorld = $state({ x: 0, y: 0 });

	let nodePositions = $state<Record<string, { x: number; y: number }>>({});
	let expandedReasoning = $state<Record<string, boolean>>({});
	let turnAssociations = $state<Record<string, TurnAssociationData>>({});
	let savingNodeId = $state<string | null>(null);

	// Lightbox state for high-res image inspection
	let lightboxImages = $state<TurnImageItem[]>([]);
	let lightboxIndex = $state(0);
	let lightboxOpen = $state(false);

	function getDefaultStarterNodes(): CustomCanvasNode[] {
		return [
			{
				color: 'slate',
				content:
					'欢迎来到 **AI 无限节点白板画布**！\n\n- **提问自动上墙**：在底部输入框发送消息，每一轮对话与 `<think>` 思考过程都会自动生成关联卡片。\n- **图片/知识/记忆一体化**：上传的题目图片会整齐对齐展示在对话节点卡片内（也可一键移至节点旁），并支持一键关联本地 **RAG 知识库** 与 **SM-2 记忆闪卡**。',
				id: 'hub-welcome',
				kind: 'hub',
				title: '✨ 你好，有什么可以帮你？（AI 知识画布中心）',
				width: 380,
				x: 60,
				y: 40
			},
			{
				color: 'sky',
				content:
					'输入关键词可直接检索后端向量知识库，并将检索到的文档段落固定在画布上、连线绑定到对话节点或注入提问框。',
				id: 'starter-rag',
				kind: 'rag',
				ragLoading: false,
				ragQuery: '',
				ragResults: [],
				title: '🔍 RAG 知识库实时检索卡',
				width: 360,
				x: 480,
				y: 40
			},
			{
				backContent:
					'SM-2 算法根据你的评分（0~5）动态更新：\n- **易度因子 (EF)**：$EF\' = EF + (0.1 - (5-q)(0.08 + (5-q)0.02))$\n- **复习间隔 (Interval)**：掌握越好，下次复习间隔呈指数级拉长。',
				color: 'violet',
				content:
					'什么是 **SM-2 间隔重复算法**？点击下方“翻转看答案”查看背面解析，或点击“拉取待复习”从后端同步今日复习题。',
				flipped: false,
				id: 'starter-flashcard',
				kind: 'flashcard',
				sm2Easiness: 2.5,
				sm2Interval: 1,
				title: '🧠 SM-2 艾宾浩斯复习闪卡',
				width: 360,
				x: 880,
				y: 40
			},
			{
				color: 'amber',
				content:
					'在此随时记录灵感、实验结论或课堂重点。\n点击右上角 **保存** 即可直接调用后端 `note_tool` 落盘为本地 Markdown 笔记。',
				id: 'starter-note',
				kind: 'note',
				title: '📝 随手便签笔记',
				width: 340,
				x: 1280,
				y: 40
			}
		];
	}

	function getDefaultStarterEdges(): CanvasEdge[] {
		return [
			{ from: 'hub-welcome', id: 'edge-hub-rag', label: '知识检索', to: 'starter-rag' },
			{ from: 'hub-welcome', id: 'edge-hub-flash', label: '记忆强化', to: 'starter-flashcard' },
			{ from: 'starter-rag', id: 'edge-rag-note', label: '摘录整理', to: 'starter-note' }
		];
	}

	let customNodes = $state<CustomCanvasNode[]>(getDefaultStarterNodes());
	let customEdges = $state<CanvasEdge[]>(getDefaultStarterEdges());

	let storageKey = $derived(
		`${STORAGE_PREFIX}${conversationsStore.activeConversation?.id ?? 'new-chat'}`
	);

	interface ChatTurnNode {
		id: string;
		turnIndex: number;
		userMessage?: DatabaseMessage;
		assistantMessage?: DatabaseMessage;
		x: number;
		y: number;
		width: number;
	}

	let activeMessagesList = $derived(
		messages && messages.length > 0 ? messages : conversationsStore.activeMessages
	);

	// Collect all images uploaded anywhere in the current conversation so follow-up turns
	// (like "我这里还有一些题目，你看一下我掌握的怎么样了" or "解析这道题") can reference them
	let allConversationImages = $derived.by<TurnImageItem[]>(() => {
		const list: TurnImageItem[] = [];

		for (const msg of activeMessagesList) {
			if (!msg.extra) continue;

			msg.extra.forEach((item: DatabaseMessageExtra, idx: number) => {
				if (item.type === AttachmentType.IMAGE && item.base64Url) {
					list.push({
						id: `conv-img-${msg.id}-${idx}`,
						name: item.name || `图片 ${list.length + 1}`,
						source: 'inherited',
						url: item.base64Url
					});
				} else if (item.type === AttachmentType.PDF && Array.isArray(item.images)) {
					item.images.forEach((pageUrl: string, pIdx: number) => {
						list.push({
							id: `conv-pdf-${msg.id}-${idx}-${pIdx}`,
							name: `${item.name || 'PDF'} (第${pIdx + 1}页)`,
							source: 'inherited',
							url: pageUrl
						});
					});
				}
			});
		}

		return list;
	});

	function estimateTurnCardHeight(turn: ChatTurnNode): number {
		let h = 240;
		const imgs = getTurnImages(turn);
		const assoc = turnAssociations[turn.id];
		const placement = assoc?.imagePlacement ?? 'top';

		if (imgs.length > 0 && placement === 'top') {
			h += 260;
		}

		const qLen = turn.userMessage?.content?.length ?? 0;
		h += Math.min(100, Math.ceil(qLen / 40) * 20);

		if (turn.assistantMessage?.reasoningContent || turn.assistantMessage?.thinking) {
			h += expandedReasoning[turn.id] ? 170 : 44;
		}

		if (assoc?.ragExpanded && assoc?.ragItems && assoc.ragItems.length > 0) {
			h += 140;
		}

		if (assoc?.memoryExpanded && assoc?.memoryItems && assoc.memoryItems.length > 0) {
			h += 140;
		}

		const aLen = turn.assistantMessage?.content?.length ?? 0;
		h += Math.min(280, Math.ceil(aLen / 50) * 16);

		return Math.min(940, Math.max(340, h));
	}

	let chatTurnNodes = $derived.by<ChatTurnNode[]>(() => {
		const pairs: { user?: DatabaseMessage; assistant?: DatabaseMessage }[] = [];
		let currentTurn: { user?: DatabaseMessage; assistant?: DatabaseMessage } | null = null;

		for (const msg of activeMessagesList) {
			if (msg.isSynthetic) continue;

			if (msg.role === MessageRole.USER) {
				if (currentTurn) {
					pairs.push(currentTurn);
				}
				currentTurn = { user: msg };
			} else if (msg.role === MessageRole.ASSISTANT) {
				if (!currentTurn) {
					currentTurn = { assistant: msg };
				} else {
					currentTurn.assistant = msg;
					pairs.push(currentTurn);
					currentTurn = null;
				}
			}
		}

		if (currentTurn) {
			pairs.push(currentTurn);
		}

		const turns: ChatTurnNode[] = [];
		const startY = customNodes.length > 0 ? 380 : 60;
		const colHeights = [startY, startY];
		const colWidth = 500;
		const colGap = 50;
		const rowGap = 36;

		pairs.forEach((pair, index) => {
			const id = `turn-${pair.user?.id ?? pair.assistant?.id ?? index}`;
			const saved = nodePositions[id];

			const col = colHeights[0] <= colHeights[1] ? 0 : 1;
			const defaultX = 60 + col * (colWidth + colGap);
			const defaultY = colHeights[col];

			const node: ChatTurnNode = {
				assistantMessage: pair.assistant,
				id,
				turnIndex: index + 1,
				userMessage: pair.user,
				width: 480,
				x: saved?.x ?? defaultX,
				y: saved?.y ?? defaultY
			};

			turns.push(node);

			const h = estimateTurnCardHeight(node);
			colHeights[col] += h + rowGap;
		});

		return turns;
	});

	function ensureTurnAssoc(turnId: string): TurnAssociationData {
		if (!turnAssociations[turnId]) {
			turnAssociations[turnId] = {
				activeImageIndex: 0,
				customImages: [],
				imagePlacement: 'top',
				memoryExpanded: false,
				memoryItems: [],
				ragExpanded: false,
				ragItems: []
			};
		}

		return turnAssociations[turnId];
	}

	// Extract all images associated with a conversation turn:
	// 1. Direct attachments on userMessage / assistantMessage
	// 2. Custom images uploaded directly to this turn node
	// 3. Images from connected 'image' custom nodes on the canvas
	// 4. Inherited conversation images when the prompt refers to problems/images
	function getTurnImages(turn: ChatTurnNode): TurnImageItem[] {
		const items: TurnImageItem[] = [];
		const seenUrls = new Set<string>();

		const pushUnique = (img: TurnImageItem) => {
			if (!img.url || seenUrls.has(img.url)) return;
			seenUrls.add(img.url);
			items.push(img);
		};

		const extractFromExtras = (extras?: DatabaseMessageExtra[], prefix = 'msg') => {
			if (!extras) return;
			extras.forEach((item, idx) => {
				if (item.type === AttachmentType.IMAGE && item.base64Url) {
					pushUnique({
						id: `${turn.id}-${prefix}-img-${idx}`,
						name: item.name || `题目图片 ${items.length + 1}`,
						source: 'message',
						url: item.base64Url
					});
				} else if (item.type === AttachmentType.PDF && Array.isArray(item.images)) {
					item.images.forEach((pageUrl, pIdx) => {
						pushUnique({
							id: `${turn.id}-${prefix}-pdf-${idx}-${pIdx}`,
							name: `${item.name || '文档'} (P${pIdx + 1})`,
							source: 'message',
							url: pageUrl
						});
					});
				}
			});
		};

		extractFromExtras(turn.userMessage?.extra, 'u');
		extractFromExtras(turn.assistantMessage?.extra, 'a');

		const assoc = turnAssociations[turn.id];

		if (assoc?.customImages) {
			for (const cImg of assoc.customImages) {
				pushUnique({
					id: cImg.id,
					name: cImg.name,
					source: 'custom',
					url: cImg.url
				});
			}
		}

		// Check connected image cards on canvas
		for (const edge of customEdges) {
			const otherId = edge.from === turn.id ? edge.to : edge.to === turn.id ? edge.from : null;

			if (!otherId) continue;

			const linkedNode = customNodes.find((n) => n.id === otherId && n.kind === 'image');

			if (linkedNode?.images && assoc?.imagePlacement !== 'side') {
				for (const nImg of linkedNode.images) {
					pushUnique({
						id: nImg.id,
						name: nImg.name,
						source: 'linked',
						url: nImg.url
					});
				}
			}
		}

		// Smart inheritance: if this turn has no direct images yet, but the user explicitly
		// enabled inheritance OR asked about "题目 / 这张图 / 这道题 / 掌握的怎么样" and the conversation has images
		const promptText = `${turn.userMessage?.content ?? ''} ${turn.assistantMessage?.content?.slice(0, 120) ?? ''}`;
		const mentionsProblemOrImage =
			/(题目|这张图|这道题|这些题|图片|上图|看下|看一下|掌握|解答|批改)/.test(promptText);

		if (
			items.length === 0 &&
			allConversationImages.length > 0 &&
			(assoc?.inheritedImages === true ||
				(assoc?.inheritedImages !== false && mentionsProblemOrImage))
		) {
			for (const convImg of allConversationImages) {
				pushUnique(convImg);
			}
		}

		return items;
	}

	function getTurnDocs(turn: ChatTurnNode): TurnDocItem[] {
		const docs: TurnDocItem[] = [];
		const extras = turn.userMessage?.extra ?? [];

		extras.forEach((item, idx) => {
			if (
				item.type === AttachmentType.TEXT ||
				item.type === AttachmentType.LEGACY_CONTEXT ||
				item.type === AttachmentType.MCP_RESOURCE ||
				item.type === AttachmentType.MCP_PROMPT
			) {
				docs.push({
					content: item.content,
					id: `${turn.id}-doc-${idx}`,
					kind: '文本附件',
					name: item.name || `附件 ${idx + 1}`
				});
			} else if (item.type === AttachmentType.PDF && item.content) {
				docs.push({
					content: item.content,
					id: `${turn.id}-pdfdoc-${idx}`,
					kind: 'PDF 文本',
					name: item.name || `PDF ${idx + 1}`
				});
			}
		});

		return docs;
	}

	function getConnectedCustomNodes(turnId: string): CustomCanvasNode[] {
		const linkedIds = new Set<string>();

		for (const edge of customEdges) {
			if (edge.from === turnId) linkedIds.add(edge.to);
			if (edge.to === turnId) linkedIds.add(edge.from);
		}

		return customNodes.filter((n) => linkedIds.has(n.id));
	}

	function cycleTurnImage(turnId: string, delta: number, total: number) {
		if (total <= 0) return;

		const assoc = ensureTurnAssoc(turnId);
		const current = assoc.activeImageIndex ?? 0;

		assoc.activeImageIndex = (current + delta + total) % total;
		persistCanvasState();
	}

	function cycleNodeImage(nodeId: string, delta: number) {
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		const imgs = customNodes[idx].images ?? [];

		if (imgs.length === 0) return;

		const current = customNodes[idx].activeImageIndex ?? 0;

		customNodes[idx].activeImageIndex = (current + delta + imgs.length) % imgs.length;
		persistCanvasState();
	}

	function triggerTurnImageUpload(turnId: string) {
		uploadTarget = { id: turnId, type: 'turn' };
		imageFileInputEl?.click();
	}

	function triggerNodeImageUpload(nodeId: string) {
		uploadTarget = { id: nodeId, type: 'node' };
		imageFileInputEl?.click();
	}

	async function handleImageInputChange(event: Event) {
		const input = event.currentTarget as HTMLInputElement;
		const files = input.files;

		if (!files || files.length === 0 || !uploadTarget) return;

		const target = uploadTarget;
		const loadedImages: Array<{ id: string; name: string; url: string }> = [];

		for (const file of Array.from(files)) {
			if (!file.type.startsWith('image/')) continue;

			const dataUrl = await new Promise<string>((resolve) => {
				const reader = new FileReader();

				reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
				reader.onerror = () => resolve('');
				reader.readAsDataURL(file);
			});

			if (dataUrl) {
				loadedImages.push({
					id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
					name: file.name,
					url: dataUrl
				});
			}
		}

		input.value = '';

		if (loadedImages.length === 0) return;

		if (target.type === 'turn') {
			const assoc = ensureTurnAssoc(target.id);

			assoc.customImages = [...(assoc.customImages ?? []), ...loadedImages];
			assoc.imagePlacement = 'top';
			assoc.activeImageIndex = (assoc.customImages.length || 1) - 1;
			toast.success(`已为该对话节点关联 ${loadedImages.length} 张图片（显示在节点上方）`);
		} else {
			const idx = customNodes.findIndex((n) => n.id === target.id);

			if (idx !== -1) {
				customNodes[idx].images = [...(customNodes[idx].images ?? []), ...loadedImages];
				customNodes[idx].activeImageIndex = (customNodes[idx].images?.length || 1) - 1;
				toast.success(`已向图片节点添加 ${loadedImages.length} 张图片`);
			}
		}

		persistCanvasState();
	}

	// Move turn images between Top Dock (Fig 2) and a Side Neighbor Card
	function moveTurnImagesToSideCard(turn: ChatTurnNode) {
		const imgs = getTurnImages(turn);

		if (imgs.length === 0) {
			toast.info('当前对话节点暂无图片，请先点击“+ 配图”上传');

			return;
		}

		const assoc = ensureTurnAssoc(turn.id);

		assoc.imagePlacement = 'side';

		const existingSideIdx = customNodes.findIndex(
			(n) => n.kind === 'image' && n.linkedTurnId === turn.id
		);

		const imagePayload = imgs.map((i) => ({ id: i.id, name: i.name, url: i.url }));

		if (existingSideIdx !== -1) {
			customNodes[existingSideIdx].images = imagePayload;
		} else {
			const newId = `imgnode-${turn.id}-${Date.now()}`;
			const sideNode: CustomCanvasNode = {
				activeImageIndex: assoc.activeImageIndex ?? 0,
				color: 'rose',
				content: `与 对话节点 #${turn.turnIndex} 关联的题目/参考图片（共 ${imagePayload.length} 张）`,
				id: newId,
				images: imagePayload,
				kind: 'image',
				linkedTurnId: turn.id,
				title: `🖼️ 对话 #${turn.turnIndex} 关联图片`,
				width: 350,
				x: Math.max(20, turn.x - 390),
				y: turn.y
			};

			customNodes = [...customNodes, sideNode];
			customEdges = [
				...customEdges,
				{
					from: newId,
					id: `edge-img-${turn.id}-${Date.now()}`,
					label: '题目配图',
					to: turn.id
				}
			];
		}

		persistCanvasState();
		toast.success('已将关联图片移至对话节点左侧并建立连线');
	}

	function dockSideImagesBackToTurnTop(imageNode: CustomCanvasNode) {
		const turnId = imageNode.linkedTurnId;

		if (!turnId) return;

		const assoc = ensureTurnAssoc(turnId);

		assoc.imagePlacement = 'top';
		assoc.activeImageIndex = imageNode.activeImageIndex ?? 0;

		if (imageNode.images && imageNode.images.length > 0) {
			const existingUrls = new Set((assoc.customImages ?? []).map((i) => i.url));
			const extra = imageNode.images.filter((i) => !existingUrls.has(i.url));

			if (extra.length > 0) {
				assoc.customImages = [...(assoc.customImages ?? []), ...extra];
			}
		}

		customNodes = customNodes.filter((n) => n.id !== imageNode.id);
		customEdges = customEdges.filter((e) => e.from !== imageNode.id && e.to !== imageNode.id);
		persistCanvasState();
		toast.success('已将图片收回至对话节点正上方（图2模式）');
	}

	// Associate RAG Knowledge directly with a Conversation Turn Node
	async function associateRagWithTurn(turn: ChatTurnNode, placeBeside = false) {
		const assoc = ensureTurnAssoc(turn.id);
		const rawQuery = (turn.userMessage?.content || turn.assistantMessage?.content || '').trim();
		const query = rawQuery.slice(0, 120);

		if (!query) {
			toast.info('当前对话节点内容为空，无法检索关联知识');

			return;
		}

		assoc.ragLoading = true;
		assoc.ragExpanded = true;

		try {
			const res = await fetch(`${base}/api/v1/rag/search`, {
				body: JSON.stringify({ query, top_k: 3 }),
				headers: { 'Content-Type': 'application/json' },
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const data = await res.json();
			const items = Array.isArray(data.results) ? data.results : [];

			const mappedRagItems = items.map(
				(r: { content?: string; text?: string; metadata?: { source?: string }; score?: number }) => ({
					content: (r.content ?? r.text ?? '').slice(0, 320),
					score: typeof r.score === 'number' ? r.score : 0,
					source: r.metadata?.source ?? '本地知识库'
				})
			);

			assoc.ragItems = mappedRagItems;

			if (placeBeside) {
				const newId = `rag-turn-${turn.turnIndex}-${Date.now()}`;
				const ragNode: CustomCanvasNode = {
					color: 'sky',
					content: `针对对话 #${turn.turnIndex} 自动关联的知识库检索结果`,
					id: newId,
					kind: 'rag',
					linkedTurnId: turn.id,
					ragLoading: false,
					ragQuery: query.slice(0, 40),
					ragResults: mappedRagItems,
					title: `📚 对话 #${turn.turnIndex} 关联知识`,
					width: 360,
					x: turn.x + turn.width + 45,
					y: turn.y
				};

				customNodes = [...customNodes, ragNode];
				customEdges = [
					...customEdges,
					{
						from: turn.id,
						id: `edge-rag-${turn.id}-${Date.now()}`,
						label: '关联知识',
						to: newId
					}
				];
				toast.success(`已在节点旁生成关联知识卡（命中 ${mappedRagItems.length} 条）`);
			} else if (mappedRagItems.length === 0) {
				toast.info('本地知识库暂未匹配到直接条目，你也可以将右侧 RAG 检索卡连线到此节点');
			} else {
				toast.success(`已为对话节点 #${turn.turnIndex} 关联 ${mappedRagItems.length} 条知识片段`);
			}
		} catch {
			toast.error('检索关联知识失败，请检查后端服务');
		} finally {
			assoc.ragLoading = false;
			persistCanvasState();
		}
	}

	// Associate Memory / SM-2 Flashcard directly with a Conversation Turn Node
	function associateMemoryWithTurn(turn: ChatTurnNode, placeBeside = false) {
		const assoc = ensureTurnAssoc(turn.id);
		const front = (turn.userMessage?.content || `对话 #${turn.turnIndex} 核心考点`).slice(0, 200);
		const back = (turn.assistantMessage?.content || '待整理要点与错题解析').slice(0, 500);

		if (placeBeside) {
			addFlashcardNode(front, back, turn.id);
			toast.success(`已在对话节点 #${turn.turnIndex} 旁生成关联 SM-2 记忆闪卡`);

			return;
		}

		const existing = assoc.memoryItems ?? [];

		assoc.memoryItems = [
			...existing,
			{
				back,
				easiness: 2.5,
				flipped: false,
				front,
				id: `mem-${Date.now()}`,
				interval: 1
			}
		];
		assoc.memoryExpanded = true;
		persistCanvasState();
		toast.success(`已在对话节点 #${turn.turnIndex} 内挂载关联记忆闪卡`);
	}

	function gradeTurnMemoryItem(turnId: string, memId: string, quality: number) {
		const assoc = ensureTurnAssoc(turnId);
		const list = assoc.memoryItems ?? [];
		const idx = list.findIndex((m) => m.id === memId);

		if (idx === -1) return;

		const item = list[idx];
		const nextEf = Math.max(
			1.3,
			Number((item.easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))).toFixed(2))
		);
		const nextInterval =
			quality < 3 ? 1 : item.interval === 1 ? 6 : Math.round(item.interval * nextEf);

		list[idx].easiness = nextEf;
		list[idx].interval = nextInterval;
		list[idx].flipped = false;
		persistCanvasState();
		toast.success(`记忆评分 ${quality} 已更新（下次间隔 ${nextInterval} 天，EF=${nextEf}）`);
	}

	function injectTurnWithAssociations(turn: ChatTurnNode) {
		const assoc = turnAssociations[turn.id];
		const imgs = getTurnImages(turn);
		const connected = getConnectedCustomNodes(turn.id);
		const parts: string[] = [];

		parts.push(`【围绕对话 #${turn.turnIndex} 继续追问】`);
		if (turn.userMessage?.content) {
			parts.push(`原问题：${turn.userMessage.content}`);
		}
		if (imgs.length > 0) {
			parts.push(`已关联题目/参考图片：${imgs.map((i) => i.name).join('、')}`);
		}
		if (assoc?.ragItems && assoc.ragItems.length > 0) {
			parts.push(
				`关联知识库参考：\n${assoc.ragItems.map((r, i) => `${i + 1}. [${r.source}] ${r.content}`).join('\n')}`
			);
		}
		if (assoc?.memoryItems && assoc.memoryItems.length > 0) {
			parts.push(
				`关联记忆考点：${assoc.memoryItems.map((m) => m.front).join('；')}`
			);
		}
		for (const node of connected) {
			parts.push(`关联卡片 [${node.title}]：${node.content.slice(0, 180)}`);
		}
		parts.push('\n我的进一步问题：');

		injectIntoChat(parts.join('\n'));
	}

	function openLightbox(images: TurnImageItem[], index: number) {
		if (images.length === 0) return;
		lightboxImages = images;
		lightboxIndex = Math.max(0, Math.min(index, images.length - 1));
		lightboxOpen = true;
	}

	// Combine automatic edges between turns + custom edges
	let allEdges = $derived.by<CanvasEdge[]>(() => {
		const result: CanvasEdge[] = [...customEdges];

		if (chatTurnNodes.length > 0) {
			const hasHub = customNodes.some((n) => n.id === 'hub-welcome');

			if (hasHub) {
				result.push({
					from: 'hub-welcome',
					id: `auto-hub-${chatTurnNodes[0].id}`,
					isAuto: true,
					label: '开启对话',
					to: chatTurnNodes[0].id
				});
			}

			for (let i = 0; i < chatTurnNodes.length - 1; i++) {
				result.push({
					from: chatTurnNodes[i].id,
					id: `auto-turn-${i}`,
					isAuto: true,
					label: `第 ${i + 2} 轮`,
					to: chatTurnNodes[i + 1].id
				});
			}
		}

		return result;
	});

	function getNodeRect(nodeId: string): { x: number; y: number; width: number; height: number } | null {
		const custom = customNodes.find((n) => n.id === nodeId);

		if (custom) {
			return { height: 220, width: custom.width, x: custom.x, y: custom.y };
		}

		const turn = chatTurnNodes.find((t) => t.id === nodeId);

		if (turn) {
			return { height: estimateTurnCardHeight(turn), width: turn.width, x: turn.x, y: turn.y };
		}

		return null;
	}

	function computeBezierPath(x1: number, y1: number, x2: number, y2: number): string {
		const dx = Math.max(Math.abs(x2 - x1) * 0.48, 60);

		return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
	}

	// Load / Save canvas state per conversation
	function loadCanvasState(key: string) {
		if (!browser) return;

		try {
			const raw = localStorage.getItem(key);

			if (!raw) {
				customNodes = getDefaultStarterNodes();
				customEdges = getDefaultStarterEdges();
				nodePositions = {};
				turnAssociations = {};

				return;
			}

			const parsed = JSON.parse(raw) as PersistedCanvasData;

			panX = typeof parsed.panX === 'number' ? parsed.panX : 80;
			panY = typeof parsed.panY === 'number' ? parsed.panY : 60;
			zoom = typeof parsed.zoom === 'number' ? parsed.zoom : 0.95;
			nodePositions = parsed.nodePositions ?? {};
			turnAssociations = parsed.turnAssociations ?? {};
			customNodes =
				Array.isArray(parsed.customNodes) && parsed.customNodes.length > 0
					? parsed.customNodes
					: getDefaultStarterNodes();
			customEdges = Array.isArray(parsed.customEdges)
				? parsed.customEdges
				: getDefaultStarterEdges();

			// Auto-detect and fix legacy overlapping positions in turn nodes (< 540px vertical distance in same col)
			if (Object.keys(nodePositions).length > 1) {
				const posEntries = Object.entries(nodePositions);
				let hasOverlap = false;

				for (let i = 0; i < posEntries.length; i++) {
					for (let j = i + 1; j < posEntries.length; j++) {
						const [idA, posA] = posEntries[i];
						const [idB, posB] = posEntries[j];

						if (idA.startsWith('turn-') && idB.startsWith('turn-')) {
							const dx = Math.abs(posA.x - posB.x);
							const dy = Math.abs(posA.y - posB.y);

							if (dx < 350 && dy < 540) {
								hasOverlap = true;
								break;
							}
						}
					}
					if (hasOverlap) break;
				}

				if (hasOverlap) {
					const cleaned: Record<string, { x: number; y: number }> = {};
					for (const [k, v] of Object.entries(nodePositions)) {
						if (!k.startsWith('turn-')) {
							cleaned[k] = v;
						}
					}
					nodePositions = cleaned;
				}
			}
		} catch {
			// Ignore malformed storage
		}
	}

	function persistCanvasState() {
		if (!browser) return;

		try {
			const payload: PersistedCanvasData = {
				customEdges,
				customNodes,
				nodePositions,
				panX,
				panY,
				turnAssociations,
				zoom
			};

			localStorage.setItem(storageKey, JSON.stringify(payload));
		} catch {
			// If base64 images exceed localStorage quota, save lightweight layout metadata
			try {
				const slimAssociations: Record<string, TurnAssociationData> = {};

				for (const [k, v] of Object.entries(turnAssociations)) {
					slimAssociations[k] = {
						...v,
						customImages: undefined
					};
				}

				const slimPayload: PersistedCanvasData = {
					customEdges,
					customNodes: customNodes.map((n) => ({ ...n, images: undefined })),
					nodePositions,
					panX,
					panY,
					turnAssociations: slimAssociations,
					zoom
				};

				localStorage.setItem(storageKey, JSON.stringify(slimPayload));
			} catch {
				// Ignore quota errors
			}
		}
	}

	let lastLoadedKey = $state('');

	$effect(() => {
		if (storageKey !== lastLoadedKey) {
			lastLoadedKey = storageKey;
			loadCanvasState(storageKey);
		}
	});

	function screenToWorld(clientX: number, clientY: number): { x: number; y: number } {
		if (!viewportEl) return { x: 0, y: 0 };

		const rect = viewportEl.getBoundingClientRect();

		return {
			x: (clientX - rect.left - panX) / zoom,
			y: (clientY - rect.top - panY) / zoom
		};
	}

	function handleCanvasMouseDown(event: MouseEvent) {
		if (event.button !== 0) return;

		const target = event.target as HTMLElement;

		if (target.closest('.canvas-node-card') || target.closest('.canvas-floating-ui')) {
			return;
		}

		isPanning = true;
		panStartMouse = { x: event.clientX, y: event.clientY };
		panStartOffset = { x: panX, y: panY };
	}

	function handleWindowMouseMove(event: MouseEvent) {
		if (isPanning) {
			panX = panStartOffset.x + (event.clientX - panStartMouse.x);
			panY = panStartOffset.y + (event.clientY - panStartMouse.y);

			return;
		}

		if (draggingNodeId) {
			const dx = (event.clientX - dragNodeStartMouse.x) / zoom;
			const dy = (event.clientY - dragNodeStartMouse.y) / zoom;
			const nextX = Math.round(dragNodeStartPos.x + dx);
			const nextY = Math.round(dragNodeStartPos.y + dy);

			const customIdx = customNodes.findIndex((n) => n.id === draggingNodeId);

			if (customIdx !== -1) {
				customNodes[customIdx].x = nextX;
				customNodes[customIdx].y = nextY;
			} else {
				nodePositions = {
					...nodePositions,
					[draggingNodeId]: { x: nextX, y: nextY }
				};
			}

			return;
		}

		if (connectingFromId) {
			connectingMouseWorld = screenToWorld(event.clientX, event.clientY);
		}
	}

	function handleWindowMouseUp() {
		if (isPanning || draggingNodeId) {
			isPanning = false;
			draggingNodeId = null;
			persistCanvasState();
		}

		if (connectingFromId) {
			connectingFromId = null;
		}
	}

	function handleWheel(event: WheelEvent) {
		const target = event.target as HTMLElement;

		if (target.closest('.canvas-scrollable-body')) {
			return;
		}

		event.preventDefault();

		if (!viewportEl) return;

		const rect = viewportEl.getBoundingClientRect();
		const cursorX = event.clientX - rect.left;
		const cursorY = event.clientY - rect.top;

		const factor = event.deltaY < 0 ? 1.08 : 0.92;
		const nextZoom = Math.min(2.0, Math.max(0.35, Number((zoom * factor).toFixed(3))));

		panX = Math.round(cursorX - ((cursorX - panX) / zoom) * nextZoom);
		panY = Math.round(cursorY - ((cursorY - panY) / zoom) * nextZoom);
		zoom = nextZoom;

		persistCanvasState();
	}

	function startDragNode(nodeId: string, currentX: number, currentY: number, event: MouseEvent) {
		if (event.button !== 0) return;

		const target = event.target as HTMLElement;

		if (
			target.closest('button') ||
			target.closest('input') ||
			target.closest('textarea') ||
			target.closest('a')
		) {
			return;
		}

		event.stopPropagation();
		draggingNodeId = nodeId;
		dragNodeStartMouse = { x: event.clientX, y: event.clientY };
		dragNodeStartPos = { x: currentX, y: currentY };
	}

	function startConnection(nodeId: string, event: MouseEvent) {
		event.stopPropagation();
		event.preventDefault();
		connectingFromId = nodeId;
		connectingMouseWorld = screenToWorld(event.clientX, event.clientY);
	}

	function completeConnection(targetNodeId: string, event: MouseEvent) {
		event.stopPropagation();

		if (!connectingFromId || connectingFromId === targetNodeId) {
			connectingFromId = null;

			return;
		}

		const exists = customEdges.some(
			(e) =>
				(e.from === connectingFromId && e.to === targetNodeId) ||
				(e.from === targetNodeId && e.to === connectingFromId)
		);

		if (!exists) {
			customEdges = [
				...customEdges,
				{
					from: connectingFromId,
					id: `edge-${Date.now()}`,
					label: '关联',
					to: targetNodeId
				}
			];
			persistCanvasState();
			toast.success('已建立节点关联连线');
		}

		connectingFromId = null;
	}

	function removeEdge(edgeId: string) {
		customEdges = customEdges.filter((e) => e.id !== edgeId);
		persistCanvasState();
	}

	// Toolbar Actions: Add nodes
	function addNoteCard(prefillTitle = '📝 新建便签笔记', prefillContent = '', linkFromId?: string) {
		const center = viewportEl
			? screenToWorld(
					viewportEl.getBoundingClientRect().left + viewportEl.clientWidth * 0.5,
					viewportEl.getBoundingClientRect().top + viewportEl.clientHeight * 0.4
				)
			: { x: 300, y: 200 };

		const offset = (customNodes.length % 5) * 28;
		const newId = `note-${Date.now()}`;
		const newNode: CustomCanvasNode = {
			color: 'amber',
			content: prefillContent || '双击或在此直接输入你的想法、摘录或课堂笔记…',
			id: newId,
			kind: 'note',
			linkedTurnId: linkFromId,
			title: prefillTitle,
			width: 340,
			x: Math.round(center.x - 150 + offset),
			y: Math.round(center.y - 100 + offset)
		};

		customNodes = [...customNodes, newNode];

		if (linkFromId) {
			customEdges = [
				...customEdges,
				{
					from: linkFromId,
					id: `edge-${Date.now()}`,
					label: '摘录便签',
					to: newId
				}
			];
		}

		persistCanvasState();
	}

	function addImageGalleryCard(linkFromId?: string) {
		const center = viewportEl
			? screenToWorld(
					viewportEl.getBoundingClientRect().left + viewportEl.clientWidth * 0.5,
					viewportEl.getBoundingClientRect().top + viewportEl.clientHeight * 0.4
				)
			: { x: 360, y: 220 };

		const newId = `image-${Date.now()}`;
		const seedImages =
			allConversationImages.length > 0
				? allConversationImages.map((i) => ({ id: i.id, name: i.name, url: i.url }))
				: [];

		const newNode: CustomCanvasNode = {
			activeImageIndex: 0,
			color: 'rose',
			content: '点击下方“+ 上传图片”添加题目或参考图，并从右侧圆点连线到任意对话节点。',
			id: newId,
			images: seedImages,
			kind: 'image',
			linkedTurnId: linkFromId,
			title: '🖼️ 题目与参考图片卡',
			width: 350,
			x: Math.round(center.x - 140),
			y: Math.round(center.y - 80)
		};

		customNodes = [...customNodes, newNode];

		if (linkFromId) {
			customEdges = [
				...customEdges,
				{
					from: newId,
					id: `edge-${Date.now()}`,
					label: '关联图片',
					to: linkFromId
				}
			];
		}

		persistCanvasState();
	}

	function addRagCard() {
		const center = viewportEl
			? screenToWorld(
					viewportEl.getBoundingClientRect().left + viewportEl.clientWidth * 0.5,
					viewportEl.getBoundingClientRect().top + viewportEl.clientHeight * 0.4
				)
			: { x: 400, y: 200 };

		const newNode: CustomCanvasNode = {
			color: 'sky',
			content: '输入检索词查询本地向量库，命中段落可直接注入对话或连线关联到对话节点。',
			id: `rag-${Date.now()}`,
			kind: 'rag',
			ragLoading: false,
			ragQuery: '',
			ragResults: [],
			title: '🔍 RAG 知识检索节点',
			width: 360,
			x: Math.round(center.x - 140),
			y: Math.round(center.y - 80)
		};

		customNodes = [...customNodes, newNode];
		persistCanvasState();
	}

	function addFlashcardNode(front = '', back = '', linkFromId?: string) {
		const center = viewportEl
			? screenToWorld(
					viewportEl.getBoundingClientRect().left + viewportEl.clientWidth * 0.5,
					viewportEl.getBoundingClientRect().top + viewportEl.clientHeight * 0.4
				)
			: { x: 450, y: 260 };

		const newId = `flash-${Date.now()}`;
		const newNode: CustomCanvasNode = {
			backContent: back || '在此编辑闪卡背面答案/解析…',
			color: 'violet',
			content: front || '在此编辑闪卡正面问题…',
			flipped: false,
			id: newId,
			kind: 'flashcard',
			linkedTurnId: linkFromId,
			sm2Easiness: 2.5,
			sm2Interval: 1,
			title: '🧠 记忆闪卡（SM-2）',
			width: 360,
			x: Math.round(center.x - 120),
			y: Math.round(center.y - 60)
		};

		customNodes = [...customNodes, newNode];

		if (linkFromId) {
			customEdges = [
				...customEdges,
				{
					from: linkFromId,
					id: `edge-${Date.now()}`,
					label: '生成闪卡',
					to: newId
				}
			];
		}

		persistCanvasState();
	}

	function deleteCustomNode(nodeId: string) {
		customNodes = customNodes.filter((n) => n.id !== nodeId);
		customEdges = customEdges.filter((e) => e.from !== nodeId && e.to !== nodeId);
		persistCanvasState();
	}

	function cycleNodeColor(nodeId: string) {
		const order: NodeCardColor[] = ['amber', 'sky', 'emerald', 'violet', 'rose', 'slate'];
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		const nextColor = order[(order.indexOf(customNodes[idx].color) + 1) % order.length];

		customNodes[idx].color = nextColor;
		persistCanvasState();
	}

	function autoLayoutNodes() {
		// Starter / custom cards layout: place neatly along the top in a grid of 3
		customNodes = customNodes.map((n, idx) => ({
			...n,
			x: 60 + (idx % 3) * 440,
			y: 40 + Math.floor(idx / 3) * 320
		}));

		const startY = customNodes.length > 0
			? 40 + Math.ceil(customNodes.length / 3) * 330 + 30
			: 50;

		// Dynamic 2-column Masonry layout for conversation turns
		const colHeights = [startY, startY];
		const colWidth = 500;
		const colGap = 50;
		const rowGap = 36;
		const nextPos: Record<string, { x: number; y: number }> = {};

		chatTurnNodes.forEach((turn) => {
			const col = colHeights[0] <= colHeights[1] ? 0 : 1;
			const x = 60 + col * (colWidth + colGap);
			const y = colHeights[col];
			nextPos[turn.id] = { x, y };

			const h = estimateTurnCardHeight(turn);
			colHeights[col] += h + rowGap;
		});

		nodePositions = nextPos;
		panX = 60;
		panY = 40;
		zoom = 0.92;
		persistCanvasState();
		toast.success('已智能重排画布节点（已优化垂直间距与防重叠）');
	}

	function resetView() {
		panX = 80;
		panY = 60;
		zoom = 0.95;
		persistCanvasState();
	}

	// Backend integrations for canvas cards
	async function executeRagSearchOnCard(nodeId: string) {
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		const query = customNodes[idx].ragQuery?.trim();

		if (!query) {
			toast.info('请输入要检索的关键词或问题');

			return;
		}

		customNodes[idx].ragLoading = true;

		try {
			const res = await fetch(`${base}/api/v1/rag/search`, {
				body: JSON.stringify({ query, top_k: 3 }),
				headers: { 'Content-Type': 'application/json' },
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const data = await res.json();
			const items = Array.isArray(data.results) ? data.results : [];

			customNodes[idx].ragResults = items.map(
				(r: { content?: string; text?: string; metadata?: { source?: string }; score?: number }) => ({
					content: (r.content ?? r.text ?? '').slice(0, 280),
					score: typeof r.score === 'number' ? r.score : 0,
					source: r.metadata?.source ?? '本地知识库'
				})
			);

			if (customNodes[idx].ragResults?.length === 0) {
				toast.info('知识库中未检索到相关段落，请先上传文档或更换关键词');
			} else {
				toast.success(`检索到 ${customNodes[idx].ragResults?.length} 条知识片段`);
			}
		} catch {
			toast.error('调用后端 RAG 检索接口失败');
		} finally {
			customNodes[idx].ragLoading = false;
			persistCanvasState();
		}
	}

	async function saveNoteToBackend(node: CustomCanvasNode) {
		savingNodeId = node.id;

		try {
			const res = await fetch(`${base}/tools`, {
				body: JSON.stringify({
					arguments: {
						action: 'create',
						content: node.content,
						title: node.title.replace(/^📝\s*/, '') || '画布便签'
					},
					name: 'note_tool'
				}),
				headers: { 'Content-Type': 'application/json' },
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			toast.success('便签已同步保存到后端 Markdown 笔记库');
		} catch {
			toast.error('保存便签到后端失败');
		} finally {
			savingNodeId = null;
		}
	}

	async function syncDueFlashcardFromBackend(nodeId: string) {
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		savingNodeId = nodeId;

		try {
			const res = await fetch(`${base}/tools`, {
				body: JSON.stringify({
					arguments: { limit: 1 },
					name: 'review_tool'
				}),
				headers: { 'Content-Type': 'application/json' },
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const data = await res.json();
			const text: string = data.content ?? '';

			if (text && !text.includes('暂无待复习')) {
				customNodes[idx].content = text;
				customNodes[idx].flipped = false;
				toast.success('已从后端拉取待复习闪卡');
			} else {
				toast.info('当前暂无到期闪卡，已为你保留示例闪卡');
			}
		} catch {
			toast.error('同步后端复习卡片失败');
		} finally {
			savingNodeId = null;
			persistCanvasState();
		}
	}

	async function gradeFlashcard(nodeId: string, quality: number) {
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		const node = customNodes[idx];
		const prevEf = node.sm2Easiness ?? 2.5;
		const prevInterval = node.sm2Interval ?? 1;
		const nextEf = Math.max(
			1.3,
			Number((prevEf + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))).toFixed(2))
		);
		const nextInterval = quality < 3 ? 1 : prevInterval === 1 ? 6 : Math.round(prevInterval * nextEf);

		customNodes[idx].sm2Easiness = nextEf;
		customNodes[idx].sm2Interval = nextInterval;
		customNodes[idx].flipped = false;
		persistCanvasState();

		toast.success(`SM-2 评分已记录（评分 ${quality} → 下次复习间隔 ${nextInterval} 天，EF=${nextEf}）`);

		if (node.cardId) {
			try {
				await fetch(`${base}/tools`, {
					body: JSON.stringify({
						arguments: { card_id: node.cardId, quality },
						name: 'grade_tool'
					}),
					headers: { 'Content-Type': 'application/json' },
					method: 'POST'
				});
			} catch {
				// Ignore background sync error
			}
		}
	}

	function injectIntoChat(text: string) {
		if (onInjectPrompt) {
			onInjectPrompt(text);
			toast.success('已填入底部提问框');
		} else {
			void copyToClipboard(text, '已复制到剪贴板');
		}
	}

	function setViewMode(mode: CanvasViewMode) {
		viewMode = mode;
		onViewModeChange?.(mode);
		if (browser) {
			try {
				localStorage.setItem('llama-preferred-view-mode', mode);
			} catch {
				// Ignore localStorage error
			}
		}
	}

	const CARD_COLOR_STYLES: Record<
		NodeCardColor,
		{ border: string; headerBg: string; badge: string; dot: string }
	> = {
		amber: {
			badge: 'bg-amber-500/15 text-amber-700 dark:text-amber-300',
			border: 'border-amber-500/35 hover:border-amber-500/60',
			dot: 'bg-amber-500',
			headerBg: 'bg-amber-500/10'
		},
		emerald: {
			badge: 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300',
			border: 'border-emerald-500/35 hover:border-emerald-500/60',
			dot: 'bg-emerald-500',
			headerBg: 'bg-emerald-500/10'
		},
		rose: {
			badge: 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
			border: 'border-rose-500/35 hover:border-rose-500/60',
			dot: 'bg-rose-500',
			headerBg: 'bg-rose-500/10'
		},
		sky: {
			badge: 'bg-sky-500/15 text-sky-700 dark:text-sky-300',
			border: 'border-sky-500/35 hover:border-sky-500/60',
			dot: 'bg-sky-500',
			headerBg: 'bg-sky-500/10'
		},
		slate: {
			badge: 'bg-primary/10 text-primary',
			border: 'border-border hover:border-primary/50',
			dot: 'bg-primary',
			headerBg: 'bg-muted/70'
		},
		violet: {
			badge: 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
			border: 'border-violet-500/35 hover:border-violet-500/60',
			dot: 'bg-violet-500',
			headerBg: 'bg-violet-500/10'
		}
	};

	onMount(() => {
		loadCanvasState(storageKey);
	});
</script>

<svelte:window onmousemove={handleWindowMouseMove} onmouseup={handleWindowMouseUp} />

<!-- Hidden file input for attaching images to any Turn Node or Image Card -->
<input
	bind:this={imageFileInputEl}
	accept="image/*"
	class="hidden"
	multiple
	onchange={handleImageInputChange}
	type="file"
/>

<!-- Top Floating Canvas Mode & Action Bar -->
<div
	class="canvas-floating-ui pointer-events-auto sticky top-2 z-30 mx-auto flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-border/70 bg-background/85 px-3 py-1.5 shadow-sm backdrop-blur-xl"
>
	<!-- Left: View Mode Switcher -->
	<div class="flex items-center gap-1 rounded-xl bg-muted/70 p-0.5">
		<button
			class="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-all {viewMode ===
			'canvas'
				? 'bg-background text-foreground shadow-xs'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => setViewMode('canvas')}
			type="button"
		>
			<LayoutGrid class="h-3.5 w-3.5" />
			<span>无限画布</span>
		</button>

		<button
			class="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-all {viewMode ===
			'split'
				? 'bg-background text-foreground shadow-xs'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => setViewMode('split')}
			type="button"
		>
			<BookOpen class="h-3.5 w-3.5" />
			<span>分屏对照</span>
		</button>

		<button
			class="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-all {viewMode ===
			'list'
				? 'bg-background text-foreground shadow-xs'
				: 'text-muted-foreground hover:text-foreground'}"
			onclick={() => setViewMode('list')}
			type="button"
		>
			<MessageSquare class="h-3.5 w-3.5" />
			<span>经典列表</span>
		</button>
	</div>

	{#if viewMode !== 'list'}
		<!-- Center: Quick Add Node Cards -->
		<div class="flex flex-wrap items-center gap-1.5">
			<Button
				class="h-7 gap-1 rounded-lg px-2.5 text-xs"
				onclick={() => addImageGalleryCard()}
				size="sm"
				variant="outline"
			>
				<ImageIcon class="h-3.5 w-3.5 text-rose-500" />
				<span>+ 图片卡</span>
			</Button>

			<Button
				class="h-7 gap-1 rounded-lg px-2.5 text-xs"
				onclick={() => addNoteCard()}
				size="sm"
				variant="outline"
			>
				<StickyNote class="h-3.5 w-3.5 text-amber-500" />
				<span>+ 便签笔记</span>
			</Button>

			<Button
				class="h-7 gap-1 rounded-lg px-2.5 text-xs"
				onclick={addRagCard}
				size="sm"
				variant="outline"
			>
				<Search class="h-3.5 w-3.5 text-sky-500" />
				<span>+ RAG 检索卡</span>
			</Button>

			<Button
				class="h-7 gap-1 rounded-lg px-2.5 text-xs"
				onclick={() => addFlashcardNode()}
				size="sm"
				variant="outline"
			>
				<Brain class="h-3.5 w-3.5 text-violet-500" />
				<span>+ SM-2 闪卡</span>
			</Button>

			<Button
				class="h-7 gap-1 rounded-lg px-2.5 text-xs"
				onclick={autoLayoutNodes}
				size="sm"
				variant="ghost"
				title="自动对齐排列所有卡片"
			>
				<Sparkles class="h-3.5 w-3.5" />
				<span class="hidden sm:inline">自动排版</span>
			</Button>
		</div>

		<!-- Right: Zoom & Viewport Reset -->
		<div class="flex items-center gap-1">
			<button
				class="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
				onclick={() => {
					zoom = Math.max(0.35, Number((zoom - 0.1).toFixed(2)));
					persistCanvasState();
				}}
				title="缩小画布"
				type="button"
			>
				<ZoomOut class="h-3.5 w-3.5" />
			</button>

			<button
				class="min-w-12 rounded-md px-1.5 py-0.5 text-center font-mono text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
				onclick={resetView}
				title="点击重置为默认缩放与视野"
				type="button"
			>
				{Math.round(zoom * 100)}%
			</button>

			<button
				class="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
				onclick={() => {
					zoom = Math.min(2.0, Number((zoom + 0.1).toFixed(2)));
					persistCanvasState();
				}}
				title="放大画布"
				type="button"
			>
				<ZoomIn class="h-3.5 w-3.5" />
			</button>

			<button
				class="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
				onclick={resetView}
				title="回到画布中心"
				type="button"
			>
				<Maximize2 class="h-3.5 w-3.5" />
			</button>
		</div>
	{/if}
</div>

{#if viewMode !== 'list'}
	<!-- Main Dot-Matrix Infinite Whiteboard Surface -->
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		bind:this={viewportEl}
		aria-label="无限节点白板画布"
		class="canvas-viewport relative mt-1.5 flex-1 h-full min-h-0 overflow-hidden rounded-2xl border border-border/80 bg-muted/20 select-none {isPanning
			? 'cursor-grabbing'
			: 'cursor-grab'}"
		style="background-image: radial-gradient(circle, color-mix(in oklab, var(--foreground) 18%, transparent) 1.25px, transparent 1.25px); background-size: {Math.max(
			10,
			Math.round(24 * zoom)
		)}px {Math.max(10, Math.round(24 * zoom))}px; background-position: {panX}px {panY}px;"
		onmousedown={handleCanvasMouseDown}
		onwheel={handleWheel}
		role="region"
	>
		<!-- Transformed World Layer -->
		<div
			class="canvas-world-layer pointer-events-none absolute top-0 left-0 origin-top-left"
			style="transform: translate3d({panX}px, {panY}px, 0) scale({zoom}); width: 4000px; height: 3000px;"
		>
			<!-- SVG Bezier Connections Layer -->
			<svg
				class="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
				xmlns="http://www.w3.org/2000/svg"
			>
				<defs>
					<marker
						id="canvas-arrow"
						markerHeight="8"
						markerWidth="8"
						orient="auto-start-reverse"
						refX="7"
						refY="4"
						viewBox="0 0 8 8"
					>
						<path d="M 0 1 L 7 4 L 0 7 z" fill="currentColor" class="text-primary/55" />
					</marker>
				</defs>

				{#each allEdges as edge (edge.id)}
					{@const fromRect = getNodeRect(edge.from)}
					{@const toRect = getNodeRect(edge.to)}
					{#if fromRect && toRect}
						{@const x1 = fromRect.x + fromRect.width}
						{@const y1 = fromRect.y + 24}
						{@const x2 = toRect.x}
						{@const y2 = toRect.y + 24}
						{@const pathD = computeBezierPath(x1, y1, x2, y2)}
						{@const midX = (x1 + x2) / 2}
						{@const midY = (y1 + y2) / 2}
						<g class="group/edge pointer-events-auto">
							<path
								class="stroke-primary/40 group-hover/edge:stroke-primary transition-colors"
								d={pathD}
								fill="none"
								marker-end="url(#canvas-arrow)"
								stroke-dasharray={edge.isAuto ? '6 4' : undefined}
								stroke-width="2.2"
							/>
							{#if edge.label}
								<g transform="translate({midX}, {midY})">
									<rect
										class="fill-background/90 stroke-border"
										height="20"
										rx="6"
										stroke-width="1"
										width="58"
										x="-29"
										y="-10"
									/>
									<text
										class="fill-muted-foreground text-[10px] font-medium"
										dominant-baseline="middle"
										text-anchor="middle"
									>
										{edge.label}
									</text>
								</g>
							{/if}
							{#if !edge.isAuto}
								<g
									class="cursor-pointer opacity-0 transition-opacity group-hover/edge:opacity-100"
									onclick={() => removeEdge(edge.id)}
									onkeydown={(e) => e.key === 'Enter' && removeEdge(edge.id)}
									role="button"
									tabindex="0"
									transform="translate({midX + 36}, {midY})"
								>
									<circle class="fill-destructive" cx="0" cy="0" r="8" />
									<text
										class="fill-destructive-foreground text-[10px] font-bold"
										dominant-baseline="middle"
										text-anchor="middle">×</text
									>
								</g>
							{/if}
						</g>
					{/if}
				{/each}

				{#if connectingFromId}
					{@const sourceRect = getNodeRect(connectingFromId)}
					{#if sourceRect}
						<path
							class="stroke-primary"
							d={computeBezierPath(
								sourceRect.x + sourceRect.width,
								sourceRect.y + 24,
								connectingMouseWorld.x,
								connectingMouseWorld.y
							)}
							fill="none"
							stroke-dasharray="5 5"
							stroke-width="2.5"
						/>
					{/if}
				{/if}
			</svg>

			<!-- 1. Custom Cards (Hub, Sticky Notes, Image Gallery Cards, RAG Search Cards, SM-2 Flashcards) -->
			{#each customNodes as node (node.id)}
				{@const style = CARD_COLOR_STYLES[node.color]}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="canvas-node-card pointer-events-auto absolute flex flex-col rounded-2xl border bg-background/95 shadow-md backdrop-blur-md transition-shadow hover:shadow-lg {style.border}"
					style="left: {node.x}px; top: {node.y}px; width: {node.width}px;"
					onmouseup={(e) => connectingFromId && completeConnection(node.id, e)}
				>
					<!-- Left Input Port -->
					<button
						aria-label="连接到此节点"
						class="absolute top-3.5 -left-2.5 z-10 flex h-5 w-5 items-center justify-center rounded-full border border-border bg-background shadow-xs transition-transform hover:scale-125"
						onmouseup={(e) => completeConnection(node.id, e)}
						title="拖入连线终点"
						type="button"
					>
						<span class="h-2 w-2 rounded-full {style.dot}"></span>
					</button>

					<!-- Right Output Port -->
					<button
						aria-label="从此节点拖出连线"
						class="absolute top-3.5 -right-2.5 z-10 flex h-5 w-5 cursor-crosshair items-center justify-center rounded-full border border-border bg-background shadow-xs transition-transform hover:scale-125"
						onmousedown={(e) => startConnection(node.id, e)}
						title="按住拖动到另一张卡片以建立连线"
						type="button"
					>
						<span class="h-2 w-2 rounded-full {style.dot}"></span>
					</button>

					<!-- Card Header (Draggable Handle) -->
					<div
						class="flex cursor-move items-center justify-between gap-2 rounded-t-2xl border-b border-border/50 px-3.5 py-2.5 {style.headerBg}"
						onmousedown={(e) => startDragNode(node.id, node.x, node.y, e)}
					>
						<input
							bind:value={node.title}
							class="min-w-0 flex-1 bg-transparent text-xs font-semibold text-foreground focus:outline-hidden"
							onchange={persistCanvasState}
							placeholder="输入卡片标题…"
						/>

						<div class="flex shrink-0 items-center gap-1">
							{#if node.kind === 'image' && node.linkedTurnId}
								<button
									class="rounded-md bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary hover:bg-primary/20"
									onclick={() => dockSideImagesBackToTurnTop(node)}
									title="将此图片画廊收回至对应对话节点的正上方（图2模式）"
									type="button"
								>
									⇡ 收回节点上方
								</button>
							{/if}

							<button
								class="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-background/60 hover:text-foreground"
								onclick={() => cycleNodeColor(node.id)}
								title="切换卡片配色"
								type="button"
							>
								<span class="h-3 w-3 rounded-full {style.dot}"></span>
							</button>

							{#if node.kind === 'note'}
								<button
									class="flex h-6 items-center gap-1 rounded-md px-1.5 text-[11px] font-medium text-muted-foreground hover:bg-background/60 hover:text-foreground"
									disabled={savingNodeId === node.id}
									onclick={() => saveNoteToBackend(node)}
									title="同步保存到后端笔记库 (note_tool)"
									type="button"
								>
									{#if savingNodeId === node.id}
										<Loader2 class="h-3 w-3 animate-spin" />
									{:else}
										<Save class="h-3 w-3" />
									{/if}
									<span>保存</span>
								</button>
							{/if}

							{#if node.id !== 'hub-welcome'}
								<button
									class="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-destructive/15 hover:text-destructive"
									onclick={() => deleteCustomNode(node.id)}
									title="删除卡片"
									type="button"
								>
									<Trash2 class="h-3.5 w-3.5" />
								</button>
							{/if}
						</div>
					</div>

					<!-- Card Body by Kind -->
					<div class="canvas-scrollable-body flex flex-col gap-2.5 p-3.5 text-xs">
						{#if node.kind === 'hub'}
							<div class="prose prose-sm dark:prose-invert max-w-none text-xs leading-relaxed">
								<MarkdownContent content={node.content} />
							</div>

							<div class="mt-1 flex flex-wrap gap-1.5 border-t border-border/40 pt-2.5">
								<Button
									class="h-7 gap-1 text-xs"
									onclick={() => injectIntoChat('请帮我总结当前知识库中的核心概念，并列出复习大纲')}
									size="sm"
									variant="secondary"
								>
									<Sparkles class="h-3 w-3" />
									<span>总结知识库大纲</span>
								</Button>

								<Button
									class="h-7 gap-1 text-xs"
									onclick={() => addNoteCard('📌 核心概念梳理', '- 概念 1：\n- 概念 2：', 'hub-welcome')}
									size="sm"
									variant="outline"
								>
									<Plus class="h-3 w-3" />
									<span>新建关联便签</span>
								</Button>
							</div>
						{:else if node.kind === 'image'}
							{@const imgs = node.images ?? []}
							{@const activeIdx = Math.min(node.activeImageIndex ?? 0, Math.max(0, imgs.length - 1))}
							{@const currentImg = imgs[activeIdx]}
							{#if currentImg}
								<div class="flex items-center justify-between gap-2.5 py-1">
									<button
										class="group flex shrink-0 flex-col items-center gap-1 transition-transform active:scale-95"
										onclick={() => cycleNodeImage(node.id, -1)}
										type="button"
									>
										<span
											class="flex h-8 w-8 items-center justify-center rounded-full border border-border/90 bg-background text-foreground shadow-xs transition-colors group-hover:border-primary group-hover:bg-primary/10 group-hover:text-primary"
										>
											<ChevronLeft class="h-4 w-4" />
										</span>
										<span class="text-[10px] font-medium text-muted-foreground group-hover:text-foreground">
											上一张
										</span>
									</button>

									<div
										class="relative flex min-h-36 max-h-52 flex-1 flex-col items-center justify-center overflow-hidden rounded-xl border border-border/80 bg-muted/30 p-1.5"
									>
										<button
											class="group/img relative flex h-full w-full items-center justify-center"
											onclick={() =>
												openLightbox(
													imgs.map((i) => ({ ...i, source: 'custom' })),
													activeIdx
												)}
											title="点击放大查看高清原图"
											type="button"
										>
											<img
												alt={currentImg.name}
												class="max-h-44 max-w-full rounded-lg object-contain transition-transform group-hover/img:scale-[1.02]"
												src={currentImg.url}
											/>
										</button>
										<div class="mt-1 flex items-center justify-between w-full px-1 text-[10px] text-muted-foreground">
											<span class="truncate max-w-36">{currentImg.name}</span>
											<span class="font-mono">{activeIdx + 1} / {imgs.length}</span>
										</div>
									</div>

									<button
										class="group flex shrink-0 flex-col items-center gap-1 transition-transform active:scale-95"
										onclick={() => cycleNodeImage(node.id, 1)}
										type="button"
									>
										<span
											class="flex h-8 w-8 items-center justify-center rounded-full border border-border/90 bg-background text-foreground shadow-xs transition-colors group-hover:border-primary group-hover:bg-primary/10 group-hover:text-primary"
										>
											<ChevronRight class="h-4 w-4" />
										</span>
										<span class="text-[10px] font-medium text-muted-foreground group-hover:text-foreground">
											下一张
										</span>
									</button>
								</div>
							{:else}
								<p class="text-[11px] leading-relaxed text-muted-foreground">
									{node.content}
								</p>
							{/if}

							<div class="flex items-center justify-between border-t border-border/40 pt-2">
								<Button
									class="h-7 gap-1 text-xs"
									onclick={() => triggerNodeImageUpload(node.id)}
									size="sm"
									variant="outline"
								>
									<Plus class="h-3 w-3" />
									<span>上传/追加图片</span>
								</Button>
								<span class="text-[10px] text-muted-foreground">
									拖右侧圆点可关联至对话
								</span>
							</div>
						{:else if node.kind === 'note'}
							<textarea
								bind:value={node.content}
								class="min-h-28 w-full resize-y rounded-lg border border-border/50 bg-background/60 p-2.5 text-xs leading-relaxed text-foreground focus:border-primary/50 focus:outline-hidden"
								onchange={persistCanvasState}
								placeholder="在此输入 Markdown 笔记内容…"
							></textarea>

							<div class="flex items-center justify-between text-[11px] text-muted-foreground">
								<span>支持拖出右侧圆点与其他卡片连线</span>
								<button
									class="font-medium text-primary hover:underline"
									onclick={() => injectIntoChat(node.content)}
									type="button"
								>
									引用到提问框 →
								</button>
							</div>
						{:else if node.kind === 'rag'}
							<form
								class="flex gap-1.5"
								onsubmit={(e) => {
									e.preventDefault();
									void executeRagSearchOnCard(node.id);
								}}
							>
								<input
									bind:value={node.ragQuery}
									class="min-w-0 flex-1 rounded-lg border border-border bg-background px-2.5 py-1.5 text-xs text-foreground focus:border-sky-500 focus:outline-hidden"
									placeholder="输入关键词检索本地 RAG 向量库…"
									type="text"
								/>
								<Button
									class="h-7 shrink-0 px-2.5 text-xs"
									disabled={node.ragLoading}
									size="sm"
									type="submit"
								>
									{#if node.ragLoading}
										<Loader2 class="h-3.5 w-3.5 animate-spin" />
									{:else}
										<Search class="h-3.5 w-3.5" />
									{/if}
								</Button>
							</form>

							{#if node.ragResults && node.ragResults.length > 0}
								<div class="max-h-48 space-y-2 overflow-y-auto pr-1">
									{#each node.ragResults as hit, i (i)}
										<div class="rounded-lg border border-border/60 bg-muted/40 p-2">
											<div class="mb-1 flex items-center justify-between text-[10px] text-muted-foreground">
												<span class="truncate font-medium">{hit.source}</span>
												<span class="font-mono">相似度 {(hit.score * 100).toFixed(0)}%</span>
											</div>
											<p class="line-clamp-3 text-[11px] leading-relaxed text-foreground/90">
												{hit.content}
											</p>
											<div class="mt-1.5 flex justify-end gap-2">
												<button
													class="text-[10px] font-medium text-sky-600 dark:text-sky-400 hover:underline"
													onclick={() =>
														addNoteCard(`📄 知识摘录：${hit.source}`, hit.content, node.id)}
													type="button"
												>
													+ 转为便签
												</button>
												<button
													class="text-[10px] font-medium text-primary hover:underline"
													onclick={() =>
														injectIntoChat(`请结合以下知识库段落回答：\n> ${hit.content}\n\n问题：`)}
													type="button"
												>
													引用提问 →
												</button>
											</div>
										</div>
									{/each}
								</div>
							{:else}
								<p class="text-[11px] leading-relaxed text-muted-foreground">
									{node.content}
								</p>
							{/if}
						{:else if node.kind === 'flashcard'}
							<div
								class="rounded-xl border border-violet-500/25 bg-violet-500/5 p-3 transition-colors"
							>
								<div class="mb-1.5 flex items-center justify-between text-[10px] text-muted-foreground">
									<span class="font-semibold uppercase">
										{node.flipped ? '🔓 背面（答案与解析）' : '❓ 正面（问题卡片）'}
									</span>
									<span class="font-mono">
										间隔: {node.sm2Interval ?? 1}d · EF: {node.sm2Easiness ?? 2.5}
									</span>
								</div>

								{#if node.flipped}
									<textarea
										bind:value={node.backContent}
										class="min-h-20 w-full resize-y bg-transparent text-xs leading-relaxed text-foreground focus:outline-hidden"
										onchange={persistCanvasState}
									></textarea>
								{:else}
									<textarea
										bind:value={node.content}
										class="min-h-20 w-full resize-y bg-transparent text-xs leading-relaxed text-foreground focus:outline-hidden"
										onchange={persistCanvasState}
									></textarea>
								{/if}
							</div>

							<div class="flex flex-wrap items-center justify-between gap-1.5">
								<Button
									class="h-7 gap-1 text-xs"
									onclick={() => {
										node.flipped = !node.flipped;
										persistCanvasState();
									}}
									size="sm"
									variant="outline"
								>
									<RotateCcw class="h-3 w-3" />
									<span>{node.flipped ? '看正面问题' : '翻转看答案'}</span>
								</Button>

								<Button
									class="h-7 gap-1 text-xs"
									disabled={savingNodeId === node.id}
									onclick={() => syncDueFlashcardFromBackend(node.id)}
									size="sm"
									variant="ghost"
								>
									<RefreshCw class="h-3 w-3 {savingNodeId === node.id ? 'animate-spin' : ''}" />
									<span>拉取待复习</span>
								</Button>
							</div>

							{#if node.flipped}
								<div class="grid grid-cols-4 gap-1 border-t border-border/40 pt-2">
									<button
										class="rounded-md bg-rose-500/15 py-1 text-[11px] font-medium text-rose-600 dark:text-rose-300 hover:bg-rose-500/25"
										onclick={() => gradeFlashcard(node.id, 1)}
										type="button"
									>
										忘记 (1)
									</button>
									<button
										class="rounded-md bg-amber-500/15 py-1 text-[11px] font-medium text-amber-600 dark:text-amber-300 hover:bg-amber-500/25"
										onclick={() => gradeFlashcard(node.id, 3)}
										type="button"
									>
										模糊 (3)
									</button>
									<button
										class="rounded-md bg-sky-500/15 py-1 text-[11px] font-medium text-sky-600 dark:text-sky-300 hover:bg-sky-500/25"
										onclick={() => gradeFlashcard(node.id, 4)}
										type="button"
									>
										掌握 (4)
									</button>
									<button
										class="rounded-md bg-emerald-500/15 py-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-300 hover:bg-emerald-500/25"
										onclick={() => gradeFlashcard(node.id, 5)}
										type="button"
									>
										熟练 (5)
									</button>
								</div>
							{/if}
						{/if}
					</div>
				</div>
			{/each}

			<!-- 2. Automatic Conversation Turn Nodes (Top Image Carousel Dock + User Prompt + Linked Knowledge/Memory + Assistant Response) -->
			{#each chatTurnNodes as turn (turn.id)}
				{@const reasoningText =
					turn.assistantMessage?.reasoningContent || turn.assistantMessage?.thinking || ''}
				{@const isReasonOpen = expandedReasoning[turn.id] ?? false}
				{@const assoc = turnAssociations[turn.id]}
				{@const turnImages = getTurnImages(turn)}
				{@const turnDocs = getTurnDocs(turn)}
				{@const connectedNodes = getConnectedCustomNodes(turn.id)}
				{@const placement = assoc?.imagePlacement ?? 'top'}
				{@const activeImgIdx = Math.min(
					assoc?.activeImageIndex ?? 0,
					Math.max(0, turnImages.length - 1)
				)}
				{@const currentTurnImg = turnImages[activeImgIdx]}

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="canvas-node-card pointer-events-auto absolute flex flex-col rounded-2xl border border-border/80 bg-background/95 shadow-md backdrop-blur-md transition-all hover:border-primary/50 hover:shadow-xl"
					style="left: {turn.x}px; top: {turn.y}px; width: {turn.width}px;"
					onmouseup={(e) => connectingFromId && completeConnection(turn.id, e)}
				>
					<!-- Left Input Port (aligned to top header) -->
					<button
						aria-label="连接到此对话节点"
						class="absolute top-3.5 -left-2.5 z-10 flex h-5 w-5 items-center justify-center rounded-full border border-border bg-background shadow-xs transition-transform hover:scale-125"
						onmouseup={(e) => completeConnection(turn.id, e)}
						title="拖入连线终点（支持将图片卡、RAG知识卡、闪卡连线关联到此对话）"
						type="button"
					>
						<span class="h-2 w-2 rounded-full bg-primary"></span>
					</button>

					<!-- Right Output Port (aligned to top header) -->
					<button
						aria-label="从此对话节点拖出连线"
						class="absolute top-3.5 -right-2.5 z-10 flex h-5 w-5 cursor-crosshair items-center justify-center rounded-full border border-border bg-background shadow-xs transition-transform hover:scale-125"
						onmousedown={(e) => startConnection(turn.id, e)}
						title="按住拖动到另一张卡片以建立连线"
						type="button"
					>
						<span class="h-2 w-2 rounded-full bg-primary"></span>
					</button>

					<!-- Unified Turn Header (Draggable) -->
					<div
						class="flex cursor-move flex-wrap items-center justify-between gap-1.5 rounded-t-2xl border-b border-border/70 bg-muted/40 px-3.5 py-2.5"
						onmousedown={(e) => startDragNode(turn.id, turn.x, turn.y, e)}
					>
						<div class="flex items-center gap-2 min-w-0">
							<span
								class="shrink-0 rounded-md bg-primary/10 px-2 py-0.5 font-mono text-[11px] font-semibold text-primary"
							>
								对话节点 #{turn.turnIndex}
							</span>
							{#if turn.assistantMessage?.model}
								<span class="max-w-36 truncate text-[11px] text-muted-foreground">
									{turn.assistantMessage.model}
								</span>
							{/if}
						</div>

						<div class="flex flex-wrap items-center gap-1">
							<button
								class="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-foreground hover:bg-accent"
								onclick={() => {
									if (turnImages.length > 0 && placement !== 'top') {
										const a = ensureTurnAssoc(turn.id);
										a.imagePlacement = 'top';
										persistCanvasState();
									} else {
										triggerTurnImageUpload(turn.id);
									}
								}}
								title="上传题目图片或展开卡片内的配图展台"
								type="button"
							>
								{turnImages.length > 0 && placement !== 'top'
									? `🖼️ 展开配图(${turnImages.length})`
									: '+ 配图'}
							</button>

							<button
								class="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-sky-600 dark:text-sky-400 hover:bg-sky-500/10"
								onclick={() => associateRagWithTurn(turn, false)}
								title="根据本轮问题自动检索并关联本地 RAG 知识库"
								type="button"
							>
								+ 关联知识
							</button>

							<button
								class="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-violet-600 dark:text-violet-400 hover:bg-violet-500/10"
								onclick={() => associateMemoryWithTurn(turn, false)}
								title="将本轮问答关联为 SM-2 记忆闪卡"
								type="button"
							>
								+ 关联记忆
							</button>

							<button
								class="rounded-md px-1.5 py-0.5 text-[11px] text-muted-foreground hover:bg-accent hover:text-foreground"
								onclick={() =>
									addNoteCard(
										`📌 摘录 #${turn.turnIndex}：${(turn.userMessage?.content ?? '').slice(0, 18)}`,
										turn.assistantMessage?.content ?? turn.userMessage?.content ?? '',
										turn.id
									)}
								title="将本轮回复摘录为旁边的独立便签卡片"
								type="button"
							>
								+ 摘录便签
							</button>

							<button
								class="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
								onclick={() =>
									copyToClipboard(
										turn.assistantMessage?.content ?? turn.userMessage?.content ?? '',
										'已复制节点内容'
									)}
								title="复制内容"
								type="button"
							>
								<Copy class="h-3 w-3" />
							</button>
						</div>
					</div>

					<!-- Integrated Image Gallery Section (Directly inside this same card container) -->
					{#if turnImages.length > 0 && placement === 'top' && currentTurnImg}
						<div class="border-b border-border/60 bg-muted/15 px-3.5 py-2.5">
							<!-- Top Mini Toolbar -->
							<div class="mb-2 flex items-center justify-between gap-2 text-[11px]">
								<div class="flex items-center gap-1.5 min-w-0">
									<span
										class="inline-flex items-center gap-1 rounded-md border border-border/70 bg-background px-1.5 py-0.5 text-[10px] font-medium text-foreground"
									>
										<ImageIcon class="h-3 w-3 text-primary" />
										<span>关联题目/图片 ({activeImgIdx + 1}/{turnImages.length})</span>
									</span>
									<span class="truncate text-[11px] text-muted-foreground" title={currentTurnImg.name}>
										{currentTurnImg.name}
									</span>
								</div>

								<div class="flex shrink-0 items-center gap-1">
									<button
										class="rounded-md px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground"
										onclick={() => triggerTurnImageUpload(turn.id)}
										title="继续添加图片到此对话节点"
										type="button"
									>
										+ 加图
									</button>
									<button
										class="rounded-md border border-border/70 bg-background px-1.5 py-0.5 text-[10px] font-medium text-foreground hover:bg-accent hover:text-primary"
										onclick={() => moveTurnImagesToSideCard(turn)}
										title="将图片拆分为对话节点旁边的独立关联卡片"
										type="button"
									>
										↗ 放至节点旁
									</button>
									<button
										class="rounded-md px-1 py-0.5 text-[10px] text-muted-foreground hover:bg-accent hover:text-foreground"
										onclick={() => {
											const a = ensureTurnAssoc(turn.id);
											a.imagePlacement = 'hidden';
											persistCanvasState();
										}}
										title="收起图片展台"
										type="button"
									>
										收起
									</button>
								</div>
							</div>

							<!-- Sleek High-Contrast Image Row: [ (<) 上一张 ]  [  图片展示框  ]  [ (>) 下一张 ] -->
							<div class="flex items-center justify-between gap-2.5">
								<!-- Left Prev Button -->
								<button
									class="group flex shrink-0 flex-col items-center gap-1 px-1 transition-transform active:scale-95"
									onclick={() => cycleTurnImage(turn.id, -1, turnImages.length)}
									title="切换到上一张图片"
									type="button"
								>
									<span
										class="flex h-8 w-8 items-center justify-center rounded-full border border-border/90 bg-background text-foreground shadow-xs transition-colors group-hover:border-primary group-hover:bg-primary/10 group-hover:text-primary"
									>
										<ChevronLeft class="h-4 w-4" />
									</span>
									<span
										class="text-[10px] font-medium text-muted-foreground group-hover:text-foreground"
									>
										上一张
									</span>
								</button>

								<!-- Center Framed Image Box -->
								<div
									class="relative flex min-h-36 max-h-56 flex-1 flex-col items-center justify-center overflow-hidden rounded-xl border border-border/80 bg-background/60 p-2 shadow-inner"
								>
									<button
										class="group/img relative flex h-full w-full items-center justify-center overflow-hidden"
										onclick={() => openLightbox(turnImages, activeImgIdx)}
										title="点击放大查看高清大图"
										type="button"
									>
										<img
											alt={currentTurnImg.name}
											class="max-h-48 max-w-full rounded-lg object-contain transition-transform duration-200 group-hover/img:scale-[1.02]"
											src={currentTurnImg.url}
										/>
										<span
											class="pointer-events-none absolute right-2 bottom-2 rounded-md bg-background/90 px-2 py-0.5 text-[10px] font-medium text-foreground opacity-0 shadow-xs backdrop-blur-xs transition-opacity group-hover/img:opacity-100"
										>
											🔍 点击放大看题
										</span>
									</button>

									{#if turnImages.length > 1}
										<div class="mt-2 flex items-center justify-center gap-1.5">
											{#each turnImages as _, dotIdx (dotIdx)}
												<button
													aria-label="切换到第 {dotIdx + 1} 张图片"
													class="h-1.5 rounded-full transition-all {dotIdx === activeImgIdx
														? 'w-4 bg-primary'
														: 'w-1.5 bg-muted-foreground/35 hover:bg-muted-foreground/60'}"
													onclick={() => {
														const a = ensureTurnAssoc(turn.id);
														a.activeImageIndex = dotIdx;
														persistCanvasState();
													}}
													type="button"
												></button>
											{/each}
										</div>
									{/if}
								</div>

								<!-- Right Next Button -->
								<button
									class="group flex shrink-0 flex-col items-center gap-1 px-1 transition-transform active:scale-95"
									onclick={() => cycleTurnImage(turn.id, 1, turnImages.length)}
									title="切换到下一张图片"
									type="button"
								>
									<span
										class="flex h-8 w-8 items-center justify-center rounded-full border border-border/90 bg-background text-foreground shadow-xs transition-colors group-hover:border-primary group-hover:bg-primary/10 group-hover:text-primary"
									>
										<ChevronRight class="h-4 w-4" />
									</span>
									<span
										class="text-[10px] font-medium text-muted-foreground group-hover:text-foreground"
									>
										下一张
									</span>
								</button>
							</div>
						</div>
					{/if}

					<!-- Turn Body -->
					<div class="canvas-scrollable-body max-h-[500px] space-y-2.5 overflow-y-auto p-3.5 text-xs">
							{#if turn.userMessage}
								<div class="rounded-xl bg-muted/70 px-3 py-2 text-xs font-medium text-foreground">
									<span class="mr-1.5 text-[10px] font-bold text-primary uppercase">提问:</span>
									{turn.userMessage.content}
								</div>
							{/if}

							<!-- ================================================================= -->
							<!-- ASSOCIATED DATA PILLS (Images / RAG Knowledge / Memory / Docs)    -->
							<!-- ================================================================= -->
							{#if turnImages.length > 0 || (assoc?.ragItems?.length ?? 0) > 0 || (assoc?.memoryItems?.length ?? 0) > 0 || turnDocs.length > 0 || connectedNodes.length > 0 || allConversationImages.length > 0}
								<div class="flex flex-wrap items-center gap-1.5 rounded-xl border border-border/60 bg-muted/30 px-2.5 py-1.5">
									<span class="flex items-center gap-1 text-[10px] font-semibold text-muted-foreground">
										<Link2 class="h-3 w-3 text-primary" />
										<span>关联数据:</span>
									</span>

									{#if turnImages.length > 0}
										<button
											class="inline-flex items-center gap-1 rounded-md bg-rose-500/15 px-2 py-0.5 text-[10px] font-medium text-rose-600 dark:text-rose-300 hover:bg-rose-500/25"
											onclick={() => {
												const a = ensureTurnAssoc(turn.id);
												a.imagePlacement = a.imagePlacement === 'top' ? 'side' : 'top';
												if (a.imagePlacement === 'side') {
													moveTurnImagesToSideCard(turn);
												} else {
													persistCanvasState();
												}
											}}
											title="点击切换图片在【节点上方（图2）】或【节点旁边】展示"
											type="button"
										>
											<ImageIcon class="h-3 w-3" />
											<span>
												图片 {turnImages.length} 张 ({placement === 'top'
													? '节点上方'
													: placement === 'side'
														? '节点旁'
														: '已折叠'})
											</span>
										</button>
									{:else if allConversationImages.length > 0}
										<button
											class="inline-flex items-center gap-1 rounded-md border border-dashed border-rose-500/40 px-2 py-0.5 text-[10px] text-rose-600 dark:text-rose-300 hover:bg-rose-500/10"
											onclick={() => {
												const a = ensureTurnAssoc(turn.id);
												a.inheritedImages = true;
												a.imagePlacement = 'top';
												persistCanvasState();
											}}
											type="button"
										>
											<ImageIcon class="h-3 w-3" />
											<span>关联会话题目图 ({allConversationImages.length}张)</span>
										</button>
									{/if}

									{#if (assoc?.ragItems?.length ?? 0) > 0 || assoc?.ragLoading}
										<button
											class="inline-flex items-center gap-1 rounded-md bg-sky-500/15 px-2 py-0.5 text-[10px] font-medium text-sky-600 dark:text-sky-300 hover:bg-sky-500/25"
											onclick={() => {
												const a = ensureTurnAssoc(turn.id);
												a.ragExpanded = !a.ragExpanded;
												persistCanvasState();
											}}
											type="button"
										>
											<Search class="h-3 w-3" />
											<span>知识库 ({assoc?.ragItems?.length ?? 0}条)</span>
										</button>
									{/if}

									{#if (assoc?.memoryItems?.length ?? 0) > 0}
										<button
											class="inline-flex items-center gap-1 rounded-md bg-violet-500/15 px-2 py-0.5 text-[10px] font-medium text-violet-600 dark:text-violet-300 hover:bg-violet-500/25"
											onclick={() => {
												const a = ensureTurnAssoc(turn.id);
												a.memoryExpanded = !a.memoryExpanded;
												persistCanvasState();
											}}
											type="button"
										>
											<Brain class="h-3 w-3" />
											<span>记忆闪卡 ({assoc?.memoryItems?.length ?? 0}张)</span>
										</button>
									{/if}

									{#each turnDocs as doc (doc.id)}
										<button
											class="inline-flex items-center gap-1 rounded-md bg-emerald-500/15 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/25"
											onclick={() => {
												const a = ensureTurnAssoc(turn.id);
												a.expandedDocId = a.expandedDocId === doc.id ? null : doc.id;
											}}
											type="button"
										>
											<FileText class="h-3 w-3" />
											<span class="max-w-28 truncate">{doc.name}</span>
										</button>
									{/each}

									{#each connectedNodes as linked (linked.id)}
										<span
											class="inline-flex items-center gap-1 rounded-md bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary"
											title="已通过画布连线关联的节点"
										>
											<span>🔗 {linked.title.slice(0, 14)}</span>
										</span>
									{/each}

									<button
										class="ml-auto rounded-md bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary hover:bg-primary/20"
										onclick={() => injectTurnWithAssociations(turn)}
										title="结合本轮关联的图片、知识库与记忆卡片继续追问"
										type="button"
									>
										💬 结合关联追问 →
									</button>
								</div>
							{/if}

							<!-- Expanded Attached Document Preview -->
							{#if assoc?.expandedDocId}
								{@const activeDoc = turnDocs.find((d) => d.id === assoc.expandedDocId)}
								{#if activeDoc}
									<div class="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-2.5">
										<div class="mb-1 flex items-center justify-between text-[10px] font-semibold text-emerald-700 dark:text-emerald-300">
											<span>📎 {activeDoc.name} ({activeDoc.kind})</span>
											<button
												class="hover:underline"
												onclick={() => {
													const a = ensureTurnAssoc(turn.id);
													a.expandedDocId = null;
												}}
												type="button"
											>
												收起
											</button>
										</div>
										<pre class="max-h-32 overflow-y-auto text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground">{activeDoc.content.slice(0, 1200)}</pre>
									</div>
								{/if}
							{/if}

							<!-- Expanded Associated RAG Knowledge Section -->
							{#if assoc?.ragExpanded}
								<div class="rounded-xl border border-sky-500/30 bg-sky-500/5 p-2.5 space-y-2">
									<div class="flex items-center justify-between text-[11px] font-medium text-sky-700 dark:text-sky-300">
										<span class="flex items-center gap-1.5">
											<Search class="h-3.5 w-3.5" />
											<span>已关联 RAG 知识库 ({assoc.ragItems?.length ?? 0} 条)</span>
										</span>
										<div class="flex items-center gap-2">
											<button
												class="text-[10px] hover:underline"
												onclick={() => associateRagWithTurn(turn, true)}
												type="button"
											>
												↗ 放至节点旁
											</button>
											<button
												class="text-[10px] hover:underline"
												onclick={() => {
													assoc.ragExpanded = false;
													persistCanvasState();
												}}
												type="button"
											>
												收起
											</button>
										</div>
									</div>

									{#if assoc.ragLoading}
										<div class="flex items-center gap-2 py-1 text-[11px] text-muted-foreground">
											<Loader2 class="h-3 w-3 animate-spin" />
											<span>正在检索与本轮对话关联的知识片段…</span>
										</div>
									{:else if assoc.ragItems && assoc.ragItems.length > 0}
										{#each assoc.ragItems as hit, rIdx (rIdx)}
											<div class="rounded-lg border border-border/50 bg-background/80 p-2">
												<div class="mb-0.5 flex items-center justify-between text-[10px] text-muted-foreground">
													<span class="font-medium truncate">{hit.source}</span>
													<span class="font-mono">{(hit.score * 100).toFixed(0)}%</span>
												</div>
												<p class="line-clamp-3 text-[11px] leading-relaxed text-foreground/90">
													{hit.content}
												</p>
											</div>
										{/each}
									{:else}
										<p class="text-[11px] text-muted-foreground">
											暂无直接命中条目，可点击右上角“↗ 放至节点旁”自定义关键词检索。
										</p>
									{/if}
								</div>
							{/if}

							<!-- Expanded Associated SM-2 Memory Section -->
							{#if assoc?.memoryExpanded && assoc.memoryItems && assoc.memoryItems.length > 0}
								<div class="rounded-xl border border-violet-500/30 bg-violet-500/5 p-2.5 space-y-2">
									<div class="flex items-center justify-between text-[11px] font-medium text-violet-700 dark:text-violet-300">
										<span class="flex items-center gap-1.5">
											<Brain class="h-3.5 w-3.5" />
											<span>已关联记忆闪卡 ({assoc.memoryItems.length} 张)</span>
										</span>
										<div class="flex items-center gap-2">
											<button
												class="text-[10px] hover:underline"
												onclick={() => associateMemoryWithTurn(turn, true)}
												type="button"
											>
												↗ 放至节点旁
											</button>
											<button
												class="text-[10px] hover:underline"
												onclick={() => {
													assoc.memoryExpanded = false;
													persistCanvasState();
												}}
												type="button"
											>
												收起
											</button>
										</div>
									</div>

									{#each assoc.memoryItems as mem (mem.id)}
										<div class="rounded-lg border border-violet-500/20 bg-background/85 p-2 space-y-1.5">
											<div class="flex items-center justify-between text-[10px] text-muted-foreground">
												<span class="font-semibold">
													{mem.flipped ? '🔓 背面解析' : '❓ 正面考点'}
												</span>
												<span class="font-mono">
													间隔 {mem.interval}d · EF {mem.easiness}
												</span>
											</div>
											<p class="text-[11px] leading-relaxed text-foreground line-clamp-3">
												{mem.flipped ? mem.back : mem.front}
											</p>
											<div class="flex items-center justify-between pt-1">
												<button
													class="text-[10px] font-medium text-violet-600 dark:text-violet-300 hover:underline"
													onclick={() => {
														mem.flipped = !mem.flipped;
														persistCanvasState();
													}}
													type="button"
												>
													{mem.flipped ? '看正面问题' : '翻转看答案'}
												</button>
												{#if mem.flipped}
													<div class="flex items-center gap-1">
														<button
															class="rounded bg-rose-500/15 px-1.5 py-0.5 text-[10px] text-rose-600"
															onclick={() => gradeTurnMemoryItem(turn.id, mem.id, 1)}
															type="button">忘记(1)</button
														>
														<button
															class="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-600"
															onclick={() => gradeTurnMemoryItem(turn.id, mem.id, 3)}
															type="button">模糊(3)</button
														>
														<button
															class="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[10px] text-emerald-600"
															onclick={() => gradeTurnMemoryItem(turn.id, mem.id, 5)}
															type="button">熟练(5)</button
														>
													</div>
												{/if}
											</div>
										</div>
									{/each}
								</div>
							{/if}

							{#if reasoningText}
								<div class="rounded-xl border border-amber-500/25 bg-amber-500/5 px-3 py-2">
									<button
										class="flex w-full items-center justify-between text-left text-[11px] font-medium text-amber-700 dark:text-amber-300"
										onclick={() =>
											(expandedReasoning = {
												...expandedReasoning,
												[turn.id]: !isReasonOpen
											})}
										type="button"
									>
										<span class="flex items-center gap-1.5">
											<Lightbulb class="h-3.5 w-3.5" />
											<span>思考过程（Reasoning）</span>
										</span>
										{#if isReasonOpen}
											<ChevronDown class="h-3.5 w-3.5" />
										{:else}
											<ChevronRight class="h-3.5 w-3.5" />
										{/if}
									</button>

									{#if isReasonOpen}
										<div
											class="mt-2 max-h-44 overflow-y-auto border-t border-amber-500/20 pt-2 text-[11px] leading-relaxed whitespace-pre-wrap text-muted-foreground"
										>
											{reasoningText}
										</div>
									{/if}
								</div>
							{/if}

							{#if turn.assistantMessage}
								<div class="prose prose-sm dark:prose-invert max-w-none text-xs leading-relaxed">
									<MarkdownContent
										attachments={turn.assistantMessage.extra}
										content={turn.assistantMessage.content || '正在生成回复…'}
									/>
								</div>
							{:else if chatStore.isLoading}
								<div class="flex items-center gap-2 py-2 text-xs text-muted-foreground">
									<Loader2 class="h-3.5 w-3.5 animate-spin text-primary" />
									<span>正在检索 RAG 知识库并生成思考回复…</span>
								</div>
							{/if}
						</div>
					</div>
			{/each}
		</div>

		<!-- Bottom-Left Canvas HUD Status Pill -->
		<div
			class="canvas-floating-ui pointer-events-auto absolute bottom-3 left-3 z-20 hidden items-center gap-3 rounded-xl border border-border/60 bg-background/80 px-3 py-1.5 text-[11px] text-muted-foreground shadow-xs backdrop-blur-md md:flex"
		>
			<span>节点总数：{customNodes.length + chatTurnNodes.length}</span>
			<span>·</span>
			<span>连线：{allEdges.length}</span>
			<span>·</span>
			<span>提示：点击对话节点「+配图 / +关联知识 / +关联记忆」可在节点上方或节点旁挂载数据</span>
		</div>
	</div>
{/if}

<!-- High-Resolution Image Lightbox Modal -->
{#if lightboxOpen && lightboxImages.length > 0}
	{@const activeLightImg = lightboxImages[lightboxIndex] ?? lightboxImages[0]}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
		onclick={() => (lightboxOpen = false)}
		onkeydown={(e) => {
			if (e.key === 'Escape') lightboxOpen = false;
			if (e.key === 'ArrowLeft')
				lightboxIndex = (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length;
			if (e.key === 'ArrowRight') lightboxIndex = (lightboxIndex + 1) % lightboxImages.length;
		}}
	>
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="relative flex max-h-[90dvh] max-w-5xl flex-col items-center gap-3 rounded-2xl border border-border/40 bg-background/95 p-4 shadow-2xl"
			onclick={(e) => e.stopPropagation()}
		>
			<div class="flex w-full items-center justify-between gap-4 border-b border-border/50 pb-2">
				<div class="flex items-center gap-2 text-xs font-medium text-foreground">
					<ImageIcon class="h-4 w-4 text-rose-500" />
					<span>{activeLightImg.name}</span>
					<span class="rounded bg-muted px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
						{lightboxIndex + 1} / {lightboxImages.length}
					</span>
				</div>
				<button
					class="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground"
					onclick={() => (lightboxOpen = false)}
					type="button"
				>
					<X class="h-4 w-4" />
				</button>
			</div>

			<div class="flex items-center gap-4">
				{#if lightboxImages.length > 1}
					<button
						class="flex flex-col items-center gap-1"
						onclick={() =>
							(lightboxIndex = (lightboxIndex - 1 + lightboxImages.length) % lightboxImages.length)}
						type="button"
					>
						<span
							class="flex h-10 w-10 items-center justify-center rounded-full border-2 border-primary/50 bg-background hover:bg-primary/10"
						>
							<ChevronLeft class="h-5 w-5" />
						</span>
						<span class="text-xs text-muted-foreground">上一张</span>
					</button>
				{/if}

				<img
					alt={activeLightImg.name}
					class="max-h-[75dvh] max-w-[78vw] rounded-xl object-contain"
					src={activeLightImg.url}
				/>

				{#if lightboxImages.length > 1}
					<button
						class="flex flex-col items-center gap-1"
						onclick={() => (lightboxIndex = (lightboxIndex + 1) % lightboxImages.length)}
						type="button"
					>
						<span
							class="flex h-10 w-10 items-center justify-center rounded-full border-2 border-primary/50 bg-background hover:bg-primary/10"
						>
							<ChevronRight class="h-5 w-5" />
						</span>
						<span class="text-xs text-muted-foreground">下一张</span>
					</button>
				{/if}
			</div>
		</div>
	</div>
{/if}
