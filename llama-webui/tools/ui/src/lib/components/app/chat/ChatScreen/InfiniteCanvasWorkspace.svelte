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
	import { MarkdownContent } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { AttachmentType, MessageRole } from '$lib/enums';
	import { chatStore, conversationsStore } from '$lib/stores';
	import type { DatabaseMessage, DatabaseMessageExtra } from '$lib/types';
	import { copyToClipboard } from '$lib/utils';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { provenanceNavigation } from '$lib/services/provenance-navigation.service';
	import { ToolsService } from '$lib/services/tools.service';
	import {
		getActiveCourseId,
		getActiveRagNamespace,
		getActiveUserId,
		learningJsonHeaders
	} from '$lib/services/learning-context';
	import {
		mapBackendHistoryToRevisions,
		mergeBackendKnowledgePoints,
		mapBackendRelationsToEdges,
		type BackendKnowledgePoint,
		type BackendRelation
	} from '$lib/services/canvas-sync';
	import { buildSiblingInfoMap } from '$lib/utils';
	import { sanitizeUtf8Text } from '$lib/utils/utf8-text';
	import type {
		CanvasEdge,
		CanvasViewMode,
		CustomCanvasNode,
		CustomNodeKind,
		KnowledgeRevision,
		NodeCardColor,
		PersistedCanvasData,
		ProvenanceAnchor,
		SemanticRelationType,
		TurnAssociationData,
		TurnDocItem,
		TurnImageItem,
		TurnMemoryItem
	} from '$lib/types/canvas';

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
	let isNavigating = $state(false);
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
	let pendingBranchSourceNodeId = $state<string | null>(null);

	const RELATION_CONFIG: Record<
		SemanticRelationType,
		{
			label: string;
			color: string;
			stroke: string;
			border: string;
			bg: string;
			marker: string;
			dash?: string;
		}
	> = {
		prerequisite: {
			bg: 'fill-sky-500/15',
			border: 'stroke-sky-500/60',
			color: 'fill-sky-700 dark:fill-sky-300',
			label: '前置知识',
			marker: 'url(#canvas-arrow-sky)',
			stroke: 'stroke-sky-500'
		},
		causal: {
			bg: 'fill-amber-500/15',
			border: 'stroke-amber-500/60',
			color: 'fill-amber-700 dark:fill-amber-300',
			label: '因果关系',
			marker: 'url(#canvas-arrow-amber)',
			stroke: 'stroke-amber-500'
		},
		similar: {
			bg: 'fill-purple-500/15',
			border: 'stroke-purple-500/60',
			color: 'fill-purple-700 dark:fill-purple-300',
			dash: '6 4',
			label: '相似概念',
			marker: 'url(#canvas-arrow-purple)',
			stroke: 'stroke-purple-500'
		},
		branch_followup: {
			bg: 'fill-emerald-500/15',
			border: 'stroke-emerald-500/60',
			color: 'fill-emerald-700 dark:fill-emerald-300',
			label: '分支追问',
			marker: 'url(#canvas-arrow-emerald)',
			stroke: 'stroke-emerald-500'
		},
		default: {
			bg: 'fill-background/90',
			border: 'stroke-border',
			color: 'fill-muted-foreground',
			label: '关联',
			marker: 'url(#canvas-arrow)',
			stroke: 'stroke-primary/40'
		}
	};

	// Lightbox state for high-res image inspection
	let lightboxImages = $state<TurnImageItem[]>([]);
	let lightboxIndex = $state(0);
	let lightboxOpen = $state(false);

	function getDefaultStarterNodes(): CustomCanvasNode[] {
		return [
			{
				color: 'slate',
				content:
					'欢迎来到 **AI 无限节点白板画布**！\n\n- **提问自动上墙**：在底部输入框发送消息，每一轮对话与 `<think>` 思考过程都会自动生成关联卡片。\n- **图片/知识/记忆一体化**：上传的图表图片会整齐对齐展示在对话节点卡片内（也可一键移至节点旁），并支持一键关联本地 **RAG 知识库** 与 **SM-2 衰减重放卡**。',
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
					'SM-2 衰减调度算法根据你的质量评分（0~5）动态更新：\n- **易度因子 (EF)**：$EF\' = EF + (0.1 - (5-q)(0.08 + (5-q)0.02))$\n- **重放间隔 (Interval)**：稳健度越高，下次重放间隔呈指数级拉长。',
				color: 'violet',
				content:
					'什么是 **SM-2 衰减重放算法**？点击下方“翻转看解析”查看背面要点，或点击“拉取待重放”从后端同步到期节点。',
				flipped: false,
				id: 'starter-flashcard',
				kind: 'flashcard',
				sm2Easiness: 2.5,
				sm2Interval: 1,
				title: '🧠 SM-2 衰减重放评估卡',
				width: 360,
				x: 880,
				y: 40
			},
			{
				color: 'amber',
				content:
					'在此随时记录灵感、实验结论或关键摘要。\n点击右上角 **保存** 即可直接调用后端 `note_tool` 落盘为本地 Markdown 笔记。',
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

	// Full message tree of the conversation (all branches) for sibling switching.
	let allConversationMessages = $state<DatabaseMessage[]>([]);
	let siblingInfoByMessageId = $derived(buildSiblingInfoMap(allConversationMessages));

	async function refreshAllConversationMessages(): Promise<void> {
		const conversation = conversationsStore.activeConversation;
		if (!conversation?.id) {
			allConversationMessages = [];
			return;
		}
		try {
			allConversationMessages = await conversationsStore.getConversationMessages(conversation.id);
		} catch (err) {
			console.warn('Failed to load conversation message tree:', err);
		}
	}

	function getTurnBranchInfo(turn: ChatTurnNode): {
		current: number;
		total: number;
	} | null {
		// Branch divergence happens at the user-message level (a follow-up question
		// becomes a sibling of the previous question); fall back to the reply id.
		const anchorId = turn.userMessage?.id ?? turn.assistantMessage?.id;
		if (!anchorId) return null;
		const info = siblingInfoByMessageId.get(anchorId);
		if (!info || info.totalSiblings <= 1) return null;
		return { current: info.currentIndex + 1, total: info.totalSiblings };
	}

	async function switchTurnBranch(turn: ChatTurnNode, delta: number): Promise<void> {
		const anchorId = turn.userMessage?.id ?? turn.assistantMessage?.id;
		if (!anchorId) return;
		const info = siblingInfoByMessageId.get(anchorId);
		if (!info || info.totalSiblings <= 1) return;
		const nextIndex = (info.currentIndex + delta + info.totalSiblings) % info.totalSiblings;
		const targetLeaf = info.siblingIds[nextIndex];
		if (!targetLeaf) return;
		await conversationsStore.navigateToSibling(targetLeaf);
		await refreshAllConversationMessages();
		toast.success(`已切换到对话分支 ${nextIndex + 1}/${info.totalSiblings}`);
	}

	$effect(() => {
		void activeMessagesList.length;
		void conversationsStore.activeConversation?.id;
		void refreshAllConversationMessages();
	});


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
						name: item.name || `输入图片 ${items.length + 1}`,
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
				content: `与 对话节点 #${turn.turnIndex} 关联的参考图片（共 ${imagePayload.length} 张）`,
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
					label: '关联配图',
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
			const res = await fetch('/api/v1/rag/search', {
				body: JSON.stringify({ query, namespace: getActiveRagNamespace(), top_k: 3 }),
				headers: learningJsonHeaders(),
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const data = await res.json();
			const items = Array.isArray(data.results) ? data.results : [];

			const mappedRagItems = items.map(
				(r: {
					content?: string;
					text?: string;
					filename?: string;
					metadata?: { source?: string; filename?: string };
					score?: number;
				}) => ({
					content: (r.content ?? r.text ?? '').slice(0, 320),
					score: typeof r.score === 'number' ? r.score : 0,
					source: r.filename ?? r.metadata?.filename ?? r.metadata?.source ?? '本地知识库'
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
		const front = (turn.userMessage?.content || `对话 #${turn.turnIndex} 核心节点`).slice(0, 200);
		const back = (turn.assistantMessage?.content || '待整理要点与结构化解析').slice(0, 500);

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
			parts.push(`已关联参考图片：${imgs.map((i) => i.name).join('、')}`);
		}
		if (assoc?.ragItems && assoc.ragItems.length > 0) {
			parts.push(
				`关联知识库参考：\n${assoc.ragItems.map((r, i) => `${i + 1}. [${r.source}] ${r.content}`).join('\n')}`
			);
		}
		if (assoc?.memoryItems && assoc.memoryItems.length > 0) {
			parts.push(
				`关联记忆节点：${assoc.memoryItems.map((m) => m.front).join('；')}`
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

	function normalizeCustomNode(rawNode: CustomCanvasNode): CustomCanvasNode {
		if (rawNode.kind === 'knowledge') {
			const provList: ProvenanceAnchor[] = Array.isArray(rawNode.provenance)
				? rawNode.provenance
				: rawNode.provenance
					? [rawNode.provenance]
					: [];

			const revList: KnowledgeRevision[] =
				Array.isArray(rawNode.revisions) && rawNode.revisions.length > 0
					? rawNode.revisions
					: [
							{
								version: 1,
								timestamp: Date.now(),
								editor: 'user',
								changeType: 'initial',
								summary: '初始版本',
								content: rawNode.content || '',
								title: rawNode.title || '',
								quoteSnippet: provList[0]?.quoteSnippet || rawNode.rawQuote || '',
								rawQuote: rawNode.rawQuote || provList[0]?.quoteSnippet || '',
								difficulty: rawNode.difficulty ?? 0.5
							}
						];

			return {
				...rawNode,
				kind: 'knowledge',
				color: rawNode.color || 'emerald',
				width: typeof rawNode.width === 'number' ? rawNode.width : 380,
				difficulty: typeof rawNode.difficulty === 'number' ? rawNode.difficulty : 0.5,
				mastery: typeof rawNode.mastery === 'number' ? rawNode.mastery : 0.3,
				currentVersion:
					typeof rawNode.currentVersion === 'number'
						? rawNode.currentVersion
						: revList[revList.length - 1]?.version ?? 1,
				provenance: provList,
				revisions: revList,
				historyExpanded: false,
				activeProvenanceIndex: 0
			};
		}
		return rawNode;
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
				provenanceNavigation.setKnowledgeNodes(customNodes);
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
					? parsed.customNodes.map(normalizeCustomNode)
					: getDefaultStarterNodes();
			customEdges = Array.isArray(parsed.customEdges)
				? parsed.customEdges
				: getDefaultStarterEdges();

			provenanceNavigation.setKnowledgeNodes(customNodes);

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
			provenanceNavigation.setKnowledgeNodes(customNodes);
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
				provenanceNavigation.setKnowledgeNodes(customNodes);
			} catch {
				// Ignore quota errors
			}
		}
	}

	let lastLoadedKey = $state('');
	let hydrateInFlight = false;

	/**
	 * Merge backend knowledge points/relations into the local canvas. Backend is the
	 * single source of truth for knowledge content/versions; local storage keeps layout.
	 */
	async function hydrateFromBackend(): Promise<void> {
		if (!browser || hydrateInFlight) return;
		const conversation = conversationsStore.activeConversation;
		if (!conversation?.id) return;

		const courseId = getActiveCourseId();
		hydrateInFlight = true;
		try {
			const headers = learningJsonHeaders();
			const [kpRes, relRes] = await Promise.all([
				fetch(`/api/courses/${encodeURIComponent(courseId)}/knowledge`, { headers }),
				fetch(`/api/courses/${encodeURIComponent(courseId)}/knowledge/relations`, { headers })
			]);
			if (!kpRes.ok) {
				const payload = (await kpRes.json().catch(() => ({}))) as { detail?: string };
				throw new Error(payload.detail || `HTTP ${kpRes.status}`);
			}
			if (!relRes.ok) {
				toast.warning('知识关系同步失败，已跳过关系连线');
			}

			const kpData = (await kpRes.json()) as { knowledge_points?: BackendKnowledgePoint[] };
			const kps = kpData.knowledge_points ?? [];
			const relData = relRes.ok
				? ((await relRes.json()) as { relations?: BackendRelation[] })
				: { relations: [] as BackendRelation[] };
			const relations = relData.relations ?? [];

			if (kps.length === 0 && relations.length === 0) return;

			const merged = mergeBackendKnowledgePoints(customNodes, kps, courseId);
			if (merged.added > 0 || merged.updated > 0) {
				customNodes = merged.nodes;
				persistCanvasState();
			}

			if (relations.length > 0) {
				const backendEdges = mapBackendRelationsToEdges(relations, customNodes);
				const existing = new Set(
					customEdges.map((e) => `${e.from}->${e.to}:${e.relType ?? 'default'}`)
				);
				const newEdges = backendEdges.filter(
					(e) => !existing.has(`${e.from}->${e.to}:${e.relType ?? 'default'}`)
				);
				if (newEdges.length > 0) {
					customEdges = [...customEdges, ...newEdges];
					persistCanvasState();
				}
			}

			if (merged.added > 0) {
				toast.info(`已从后端补齐 ${merged.added} 个知识节点`);
			}

			// Retry knowledge nodes whose earlier backend write failed.
			for (const node of customNodes.filter(
				(n) => n.kind === 'knowledge' && n.syncPending
			)) {
				void persistKnowledgeNode(node, 'retry_sync');
			}
		} catch (err) {
			console.warn('Canvas backend hydration skipped:', err);
			toast.error(
				`无法从后端同步知识画布：${err instanceof Error ? err.message : String(err)}`
			);
		} finally {
			hydrateInFlight = false;
		}
	}

	$effect(() => {
		if (storageKey !== lastLoadedKey) {
			lastLoadedKey = storageKey;
			loadCanvasState(storageKey);
			if (conversationsStore.activeConversation?.id) {
				void hydrateFromBackend();
			}
		}
	});

	$effect(() => {
		provenanceNavigation.setKnowledgeNodes(customNodes);
	});

	function focusCanvasNode(nodeId: string) {
		if (!nodeId) return;

		let targetX = 0;
		let targetY = 0;
		let targetWidth = 380;
		let targetHeight = 260;

		const cNode = customNodes.find((n) => n.id === nodeId);
		if (cNode) {
			targetX = cNode.x;
			targetY = cNode.y;
			targetWidth = cNode.width;
		} else {
			const turn = chatTurnNodes.find((t) => t.id === nodeId || t.assistantMessage?.id === nodeId);
			if (turn) {
				targetX = turn.x;
				targetY = turn.y;
				targetWidth = turn.width;
				targetHeight = estimateTurnCardHeight(turn);
			} else {
				return;
			}
		}

		const centerX = targetX + targetWidth / 2;
		const centerY = targetY + targetHeight / 2;

		if (zoom < 0.8) {
			zoom = 0.95;
		}

		const vpWidth = viewportEl?.clientWidth || 1000;
		const vpHeight = viewportEl?.clientHeight || 700;

		panX = Math.round(vpWidth / 2 - centerX * zoom);
		panY = Math.round(vpHeight / 2 - centerY * zoom);

		isNavigating = true;
		setTimeout(() => {
			isNavigating = false;
		}, 480);
	}

	function buildProvenanceDict(node: CustomCanvasNode): Record<string, unknown> | undefined {
		const anchors = node.provenance ?? [];
		if (anchors.length === 0) return undefined;
		const primary = anchors[0];
		return {
			message_id: primary.messageId,
			turn_index: primary.turnIndex,
			doc_id: primary.docId,
			chunk_id: primary.chunkId,
			doc_title: primary.docTitle,
			page_start: primary.pageStart,
			page_end: primary.pageEnd,
			quote_snippet: primary.quoteSnippet,
			extra: {
				all_anchors: anchors.map((a) => ({
					id: a.id,
					message_id: a.messageId,
					turn_index: a.turnIndex,
					doc_id: a.docId,
					chunk_id: a.chunkId,
					doc_title: a.docTitle,
					page_start: a.pageStart,
					page_end: a.pageEnd,
					quote_snippet: a.quoteSnippet,
					created_at: a.createdAt
				}))
			}
		};
	}

	/** Adopt backend knowledge-point state (backend is the single source of versions). */
	function adoptBackendKnowledge(node: CustomCanvasNode, kp: BackendKnowledgePoint): void {
		if (!kp?.id) return;
		node.knowledgePointId = kp.id;
		node.courseId = kp.course_id || node.courseId || getActiveCourseId();
		node.currentVersion = Number(kp.version) || node.currentVersion || 1;
		const history = mapBackendHistoryToRevisions(kp.history, kp.name);
		if (history.length > 0) node.revisions = history;
		node.syncPending = false;
	}

	/** Persist a knowledge card to the backend, adopting the returned version/history. */
	async function persistKnowledgeNode(node: CustomCanvasNode, changeType: string): Promise<boolean> {
		savingNodeId = node.id;
		const courseId = node.courseId || getActiveCourseId();
		try {
			const res = await fetch(`/api/courses/${encodeURIComponent(courseId)}/knowledge`, {
				method: 'POST',
				headers: learningJsonHeaders(),
				body: JSON.stringify({
					id: node.knowledgePointId,
					user_id: getActiveUserId(),
					name: node.title,
					summary: node.title,
					difficulty: node.difficulty ?? 0.5,
					raw_quote: node.rawQuote || node.content,
					content: node.content,
					provenance: buildProvenanceDict(node),
					prereq_ids: [],
					change_type: changeType
				})
			});
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const data = await res.json();
			const kp = (data.knowledge_point ?? data) as BackendKnowledgePoint;
			adoptBackendKnowledge(node, kp);
			return true;
		} catch (err) {
			console.warn('Knowledge node backend sync failed:', err);
			node.syncPending = true;
			return false;
		} finally {
			savingNodeId = null;
			persistCanvasState();
		}
	}

	async function handleCreateKnowledgeNode(detail: {
		title: string;
		quoteSnippet: string;
		messageId?: string;
		turnIndex?: number;
		courseId?: string;
		docId?: string;
		chunkId?: string;
		docTitle?: string;
		pageStart?: number;
		pageEnd?: number;
		difficulty?: number;
		mastery?: number;
	}) {
		const originTurn = chatTurnNodes.find(
			(t) =>
				(detail.messageId && t.assistantMessage?.id === detail.messageId) ||
				(detail.turnIndex && t.turnIndex === detail.turnIndex)
		);

		let targetX: number;
		let targetY: number;

		if (originTurn) {
			targetX = originTurn.x + originTurn.width + 50;
			const siblings = customNodes.filter(
				(n) =>
					n.linkedTurnId === originTurn.id ||
					n.provenance?.some((p) => p.messageId === detail.messageId)
			);
			targetY = originTurn.y + siblings.length * 380;
		} else {
			const center = screenToWorld(
				(viewportEl?.clientWidth || 1000) / 2,
				(viewportEl?.clientHeight || 700) / 2
			);
			targetX = Math.round(center.x - 190 + (customNodes.length % 5) * 24);
			targetY = Math.round(center.y - 140 + (customNodes.length % 5) * 24);
		}

		const anchor: ProvenanceAnchor = {
			id: `prov-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
			messageId: detail.messageId,
			turnIndex: detail.turnIndex,
			docId: detail.docId,
			chunkId: detail.chunkId,
			docTitle: detail.docTitle,
			pageStart: detail.pageStart,
			pageEnd: detail.pageEnd,
			quoteSnippet: detail.quoteSnippet,
			createdAt: Date.now()
		};

		const revision: KnowledgeRevision = {
			version: 1,
			timestamp: Date.now(),
			editor: 'extract',
			changeType: 'initial',
			summary: detail.turnIndex ? `初始从对话 #${detail.turnIndex} 提炼` : '初始划词提炼',
			content: detail.quoteSnippet,
			title: detail.title,
			quoteSnippet: detail.quoteSnippet,
			rawQuote: detail.quoteSnippet,
			difficulty: detail.difficulty ?? 0.5
		};

		const newNodeId = `kn-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
		const newCard: CustomCanvasNode = {
			id: newNodeId,
			kind: 'knowledge',
			courseId: detail.courseId || getActiveCourseId(),
			syncPending: true,
			title: detail.title,
			content: detail.quoteSnippet,
			rawQuote: detail.quoteSnippet,
			color: 'emerald',
			width: 380,
			x: targetX,
			y: targetY,
			difficulty: detail.difficulty ?? 0.5,
			mastery: detail.mastery ?? 0.3,
			currentVersion: 1,
			linkedTurnId: originTurn?.id,
			provenance: [anchor],
			revisions: [revision],
			historyExpanded: false,
			activeProvenanceIndex: 0
		};

		customNodes = [...customNodes, newCard];

		if (originTurn) {
			customEdges = [
				...customEdges,
				{
					id: `edge-extract-${Date.now()}`,
					from: originTurn.id,
					to: newNodeId,
					label: '知识提炼',
					isAuto: true
				}
			];
		}

		persistCanvasState();
		focusCanvasNode(newNodeId);

		const created = customNodes.find((n) => n.id === newNodeId);
		if (created) {
			const ok = await persistKnowledgeNode(created, 'create');
			toast[ok ? 'success' : 'warning'](
				ok ? `已保存知识节点「${created.title}」` : `知识节点「${created.title}」已本地暂存，后端同步稍后重试`
			);
		}
	}

	async function handleAppendKnowledgeNode(detail: {
		targetNodeId: string;
		courseId?: string;
		quoteSnippet: string;
		messageId?: string;
		turnIndex?: number;
		docId?: string;
		chunkId?: string;
		docTitle?: string;
		pageStart?: number;
		pageEnd?: number;
	}) {
		const idx = customNodes.findIndex((n) => n.id === detail.targetNodeId);
		if (idx === -1) return;

		const target = customNodes[idx];
		const anchor: ProvenanceAnchor = {
			id: `prov-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
			messageId: detail.messageId,
			turnIndex: detail.turnIndex,
			docId: detail.docId,
			chunkId: detail.chunkId,
			docTitle: detail.docTitle,
			pageStart: detail.pageStart,
			pageEnd: detail.pageEnd,
			quoteSnippet: detail.quoteSnippet,
			createdAt: Date.now()
		};

		const nextVersion = (target.currentVersion ?? 1) + 1;
		const revision: KnowledgeRevision = {
			version: nextVersion,
			timestamp: Date.now(),
			editor: 'extract',
			changeType: 'append_quote',
			summary: detail.turnIndex ? `追加来自对话 #${detail.turnIndex} 的引用` : '追加引用片段',
			content: target.content,
			title: target.title,
			quoteSnippet: detail.quoteSnippet,
			rawQuote: detail.quoteSnippet,
			difficulty: target.difficulty ?? 0.5
		};

		target.rawQuote = target.rawQuote
			? `${target.rawQuote}\n\n---\n\n${detail.quoteSnippet}`
			: detail.quoteSnippet;
		target.provenance = [...(target.provenance ?? []), anchor];
		target.revisions = [...(target.revisions ?? []), revision];
		target.currentVersion = nextVersion;
		target.activeProvenanceIndex = (target.provenance.length || 1) - 1;
		if (detail.courseId) target.courseId = detail.courseId;

		persistCanvasState();
		focusCanvasNode(target.id);

		const ok = await persistKnowledgeNode(target, 'append_quote');
		toast[ok ? 'success' : 'warning'](
			ok ? `已追加引用到「${target.title}」并同步后端` : `引用已本地追加，「${target.title}」后端同步稍后重试`
		);
	}

	function handleKnowledgeNodeBlur(node: CustomCanvasNode) {
		const lastRev = node.revisions?.[node.revisions.length - 1];
		if (lastRev && lastRev.content === node.content && lastRev.title === node.title) {
			persistCanvasState();
			return;
		}

		const nextVersion = (node.currentVersion ?? 1) + 1;
		const revision: KnowledgeRevision = {
			version: nextVersion,
			timestamp: Date.now(),
			editor: 'user',
			changeType: 'edit_content',
			summary: '更新概念笔记与思考',
			content: node.content,
			title: node.title,
			difficulty: node.difficulty ?? 0.5
		};

		node.revisions = [...(node.revisions ?? []), revision];
		node.currentVersion = nextVersion;
		persistCanvasState();

		void persistKnowledgeNode(node, 'edit_content');
	}

	async function handleKnowledgeRollback(node: CustomCanvasNode, targetRev: KnowledgeRevision) {
		node.title = targetRev.title ?? node.title;
		node.content = targetRev.content;
		if (targetRev.difficulty !== undefined) {
			node.difficulty = targetRev.difficulty;
		}

		const nextVersion = (node.currentVersion ?? 1) + 1;
		const rollbackRev: KnowledgeRevision = {
			version: nextVersion,
			timestamp: Date.now(),
			editor: 'user',
			changeType: 'revert',
			summary: `回退至版本 v${targetRev.version}`,
			content: node.content,
			title: node.title,
			difficulty: node.difficulty ?? 0.5
		};

		node.revisions = [...(node.revisions ?? []), rollbackRev];
		node.currentVersion = nextVersion;
		node.historyExpanded = false;
		persistCanvasState();

		const courseId = node.courseId || getActiveCourseId();
		if (node.knowledgePointId) {
			try {
				const res = await fetch(
					`/api/courses/${encodeURIComponent(courseId)}/knowledge/${encodeURIComponent(node.knowledgePointId)}/rollback`,
					{
						method: 'POST',
						headers: learningJsonHeaders(),
						body: JSON.stringify({ target_version: targetRev.version })
					}
				);
				if (res.ok) {
					const data = await res.json();
					adoptBackendKnowledge(node, (data.knowledge_point ?? data) as BackendKnowledgePoint);
					persistCanvasState();
					toast.success(`已回退知识节点「${node.title}」至 v${targetRev.version}`);
					return;
				}
				console.warn('Backend rollback rejected:', res.status);
			} catch (err) {
				console.warn('Backend rollback sync warning:', err);
			}
		}

		// Backend offline or version mismatch: persist the rolled-back content as a new edit.
		node.syncPending = true;
		const ok = await persistKnowledgeNode(node, `rollback_to_v${targetRev.version}`);
		toast[ok ? 'success' : 'warning'](
			ok
				? `已回退知识节点「${node.title}」至 v${targetRev.version}`
				: `已本地回退「${node.title}」至 v${targetRev.version}，后端同步稍后重试`
		);
	}
	function convertKnowledgeToFlashcard(node: CustomCanvasNode) {
		const newFlashcardId = `flash-${Date.now()}`;
		const quote = node.provenance?.[0]?.quoteSnippet || node.rawQuote || '';
		const back = quote ? `${node.content}\n\n---\n**原文出处**：\n> ${quote}` : node.content;

		const flashCard: CustomCanvasNode = {
			id: newFlashcardId,
			kind: 'flashcard',
			title: `🧠 闪卡：${node.title}`,
			content: `什么是「${node.title}」？请尝试回忆其核心定义与要点。`,
			backContent: back,
			color: 'violet',
			width: 360,
			x: node.x + node.width + 40,
			y: node.y,
			flipped: false,
			cardId: node.knowledgePointId,
			sm2Interval: 1,
			sm2Easiness: 2.5
		};

		customNodes = [...customNodes, flashCard];
		customEdges = [
			...customEdges,
			{
				id: `edge-kn-flash-${Date.now()}`,
				from: node.id,
				to: newFlashcardId,
				label: '记忆强化',
				isAuto: true
			}
		];
		persistCanvasState();
		toast.success(`已生成重放评估卡并与「${node.title}」关联`);
	}

	onMount(() => {
		provenanceNavigation.setKnowledgeNodes(customNodes);

		const handleCreateNodeEvent = (e: Event) => {
			const customEvent = e as CustomEvent;
			handleCreateKnowledgeNode(customEvent.detail);
		};
		const handleAppendNodeEvent = (e: Event) => {
			const customEvent = e as CustomEvent;
			handleAppendKnowledgeNode(customEvent.detail);
		};
		const handleFocusNodeEvent = (e: Event) => {
			const customEvent = e as CustomEvent<{ nodeId: string }>;
			focusCanvasNode(customEvent.detail.nodeId);
		};

		document.addEventListener('canvas:create-knowledge-node', handleCreateNodeEvent);
		document.addEventListener('canvas:append-to-knowledge-node', handleAppendNodeEvent);
		document.addEventListener('provenance-focus-canvas-node', handleFocusNodeEvent);

		return () => {
			document.removeEventListener('canvas:create-knowledge-node', handleCreateNodeEvent);
			document.removeEventListener('canvas:append-to-knowledge-node', handleAppendNodeEvent);
			document.removeEventListener('provenance-focus-canvas-node', handleFocusNodeEvent);
		};
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

	async function syncRelationToBackend(edge: CanvasEdge, isDelete = false) {
		const fromNode = customNodes.find((n) => n.id === edge.from);
		const toNode = customNodes.find((n) => n.id === edge.to);

		const srcId = fromNode?.knowledgePointId;
		const dstId = toNode?.knowledgePointId;
		const relType = edge.relType;
		const courseId = fromNode?.courseId || getActiveCourseId();

		if (!relType) return;
		// Only sync edges between persisted knowledge points; title/local ids would
		// create dangling garbage rows in the backend graph.
		if (!srcId || !dstId) {
			if (!isDelete) {
				toast.info('两端都需为已保存的知识节点，才能同步语义关系到后端');
			}
			return;
		}

		try {
			const res = isDelete
				? await fetch(
						`/api/courses/${encodeURIComponent(courseId)}/knowledge/relations?src_id=${encodeURIComponent(srcId)}&rel_type=${encodeURIComponent(relType)}&dst_id=${encodeURIComponent(dstId)}`,
						{ method: 'DELETE', headers: learningJsonHeaders() }
					)
				: await fetch(`/api/courses/${encodeURIComponent(courseId)}/knowledge/relations`, {
						body: JSON.stringify({
							dst_id: dstId,
							metadata: { edge_id: edge.id },
							rel_type: relType,
							src_id: srcId
						}),
						headers: learningJsonHeaders(),
						method: 'POST'
					});

			if (!res.ok) {
				const payload = (await res.json().catch(() => ({}))) as { detail?: string };
				throw new Error(payload.detail || `HTTP ${res.status}`);
			}
		} catch (err) {
			toast.error(
				`${isDelete ? '删除' : '保存'}知识关系失败：${
					err instanceof Error ? err.message : String(err)
				}`
			);
		}
	}

	async function cycleOrSetEdgeRelation(edgeId: string) {
		const idx = customEdges.findIndex((e) => e.id === edgeId);
		if (idx === -1) return;

		const current = (customEdges[idx].relType || 'default') as SemanticRelationType;
		const types: SemanticRelationType[] = [
			'prerequisite',
			'causal',
			'similar',
			'branch_followup',
			'default'
		];

		const targetType = types[(types.indexOf(current) + 1) % types.length];
		const cfg = RELATION_CONFIG[targetType];

		customEdges[idx] = {
			...customEdges[idx],
			label: cfg.label,
			relType: targetType === 'default' ? undefined : (targetType as CanvasEdge['relType'])
		};

		persistCanvasState();
		toast.success(`连线关系已切换为：【${cfg.label}】`);
		void syncRelationToBackend(customEdges[idx]);
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
			const fromNode = customNodes.find((n) => n.id === connectingFromId);
			const toNode = customNodes.find((n) => n.id === targetNodeId);
			const isKnowledge = fromNode?.kind === 'knowledge' || toNode?.kind === 'knowledge';
			const initialRelType: SemanticRelationType = isKnowledge ? 'prerequisite' : 'default';
			const cfg = RELATION_CONFIG[initialRelType];

			const newEdge: CanvasEdge = {
				from: connectingFromId,
				id: `edge-${Date.now()}`,
				label: cfg.label,
				relType: initialRelType === 'default' ? undefined : (initialRelType as CanvasEdge['relType']),
				to: targetNodeId
			};

			customEdges = [...customEdges, newEdge];
			persistCanvasState();
			toast.success(`已建立连线：【${cfg.label}】（点击连线标签可切换类型）`);
			void syncRelationToBackend(newEdge);
		}

		connectingFromId = null;
	}

	function removeEdge(edgeId: string) {
		const edge = customEdges.find((e) => e.id === edgeId);
		if (edge) {
			void syncRelationToBackend(edge, true);
		}
		customEdges = customEdges.filter((e) => e.id !== edgeId);
		persistCanvasState();
	}

	async function handleKnowledgeBranchFollowup(node: CustomCanvasNode) {
		const provList = node.provenance ?? [];
		const activeProvIdx = Math.min(
			node.activeProvenanceIndex ?? 0,
			Math.max(0, provList.length - 1)
		);
		const anchor = provList[activeProvIdx] || provList[0];
		if (anchor?.messageId) {
			try {
				await conversationsStore.updateCurrentNode(anchor.messageId);
				await conversationsStore.refreshActiveMessages();
			} catch (e) {
				console.warn('Branch switch error', e);
			}
		}

		const docRef = anchor?.docTitle
			? `[出处: ${anchor.docTitle}${anchor.pageStart ? ` P.${anchor.pageStart}` : ''}]`
			: '';
		const excerpt = node.rawQuote || anchor?.quoteSnippet || '';
		const prompt = `【基于知识节点「${node.title}」展开分支追问】\n${docRef ? `${docRef}\n` : ''}${excerpt ? `> 依据摘录：${excerpt.slice(0, 160)}\n` : ''}- 核心要点：${node.content.slice(0, 180)}\n\n我的追问：`;

		injectIntoChat(prompt);
		pendingBranchSourceNodeId = node.id;
		toast.success(
			`已锚定至「${node.title}」所处对话分支（原分支完整保留），发送提问后将自动生成独立对话分支！`
		);
	}

	async function handleTurnBranchFollowup(turn: ChatTurnNode) {
		const anchorId = turn.assistantMessage?.id || turn.userMessage?.id;
		if (anchorId) {
			try {
				await conversationsStore.updateCurrentNode(anchorId);
				await conversationsStore.refreshActiveMessages();
			} catch (e) {
				console.warn('Turn branch switch error', e);
			}
		}
		const prompt = `【围绕对话 #${turn.turnIndex} 开启新分支追问】\n原问题：${turn.userMessage?.content || ''}\n进一步追问：`;
		injectIntoChat(prompt);
		pendingBranchSourceNodeId = turn.id;
		toast.success(`已锚定至对话 #${turn.turnIndex} 历史分支（后续原分支完整保留）！`);
	}

	$effect(() => {
		if (pendingBranchSourceNodeId && chatTurnNodes.length > 0) {
			const latestTurn = chatTurnNodes[chatTurnNodes.length - 1];
			if (latestTurn.id !== pendingBranchSourceNodeId) {
				const exists = customEdges.some(
					(e) => e.from === pendingBranchSourceNodeId && e.to === latestTurn.id
				);
				if (!exists) {
					const newEdge: CanvasEdge = {
						from: pendingBranchSourceNodeId,
						id: `edge-branch-${Date.now()}`,
						label: '分支追问',
						relType: 'branch_followup',
						to: latestTurn.id
					};
					customEdges = [...customEdges, newEdge];
					persistCanvasState();
					void syncRelationToBackend(newEdge);
				}
				pendingBranchSourceNodeId = null;
			}
		}
	});

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
			content: prefillContent || '双击或在此直接输入你的想法、摘录或结构化笔记…',
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
			content: '点击下方“+ 上传图片”添加图表或参考图，并从右侧圆点连线到任意对话节点。',
			id: newId,
			images: seedImages,
			kind: 'image',
			linkedTurnId: linkFromId,
			title: '🖼️ 视觉与参考图片卡',
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

	async function deleteCustomNode(nodeId: string) {
		const node = customNodes.find((n) => n.id === nodeId);
		if (!node) return;

		if (node.kind === 'knowledge' && node.knowledgePointId) {
			if (!confirm(`确定删除知识节点「${node.title}」吗？后端知识节点与关联状态记录将一并级联删除。`)) {
				return;
			}
			const courseId = node.courseId || getActiveCourseId();
			try {
				const res = await fetch(
					`/api/courses/${encodeURIComponent(courseId)}/knowledge/${encodeURIComponent(node.knowledgePointId)}`,
					{ method: 'DELETE', headers: learningJsonHeaders() }
				);
				if (!res.ok && res.status !== 404) {
					toast.error(`后端删除失败（HTTP ${res.status}），节点已保留`);
					return;
				}
			} catch (err) {
				console.warn('Backend knowledge delete failed:', err);
				toast.error('后端删除失败，节点已保留');
				return;
			}
		}

		customNodes = customNodes.filter((n) => n.id !== nodeId);
		customEdges = customEdges.filter((e) => e.from !== nodeId && e.to !== nodeId);
		persistCanvasState();
		toast.success('已删除画布节点');
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
			const res = await fetch('/api/v1/rag/search', {
				body: JSON.stringify({ query, namespace: getActiveRagNamespace(), top_k: 3 }),
				headers: learningJsonHeaders(),
				method: 'POST'
			});

			if (!res.ok) throw new Error(`HTTP ${res.status}`);

			const data = await res.json();
			const items = Array.isArray(data.results) ? data.results : [];

			customNodes[idx].ragResults = items.map(
				(r: {
					content?: string;
					text?: string;
					filename?: string;
					metadata?: { source?: string; filename?: string };
					score?: number;
				}) => ({
					content: (r.content ?? r.text ?? '').slice(0, 280),
					score: typeof r.score === 'number' ? r.score : 0,
					source: r.filename ?? r.metadata?.filename ?? r.metadata?.source ?? '本地知识库'
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
			const result = await ToolsService.executeTool('note_tool', {
				action: 'create',
				content: node.content,
				title: node.title.replace(/^📝\s*/, '') || '画布便签'
			});

			if (result.isError) throw new Error(result.content);

			toast.success(`便签已保存到后端笔记库（${result.content}）`);
		} catch (err) {
			toast.error(
				`保存便签到后端失败：${err instanceof Error ? err.message : String(err)}`
			);
		} finally {
			savingNodeId = null;
		}
	}

	async function syncDueFlashcardFromBackend(nodeId: string) {
		const idx = customNodes.findIndex((n) => n.id === nodeId);

		if (idx === -1) return;

		savingNodeId = nodeId;

		try {
			const result = await ToolsService.executeTool('review_tool', { action: 'due' });

			if (result.isError) throw new Error(result.content);

			const text = result.content;

			if (text && !text.includes('暂无到期')) {
				customNodes[idx].content = text;
				customNodes[idx].flipped = false;
				toast.success('已从后端拉取待重放评估卡');
			} else {
				toast.info('当前暂无到期闪卡，已为你保留示例闪卡');
			}
		} catch (err) {
			toast.error(
				`同步后端重放卡片失败：${err instanceof Error ? err.message : String(err)}`
			);
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

		if (!node.cardId) {
			toast.success(`SM-2 评分已记录（评分 ${quality} → 下次重放间隔 ${nextInterval} 天，EF=${nextEf}）`);

			return;
		}

		try {
			const result = await ToolsService.executeTool('review_tool', {
				action: 'grade',
				kp_id: node.cardId,
				quality
			});

			if (result.isError) throw new Error(result.content);

			toast.success(
				`SM-2 评分已记录并同步后端（评分 ${quality} → 下次重放间隔 ${nextInterval} 天，EF=${nextEf}）`
			);
		} catch (err) {
			toast.warning(
				`本地已记录评分，但后端同步失败：${err instanceof Error ? err.message : String(err)}`
			);
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
			class:canvas-world-layer--navigating={isNavigating}
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
					<marker
						id="canvas-arrow-sky"
						markerHeight="8"
						markerWidth="8"
						orient="auto-start-reverse"
						refX="7"
						refY="4"
						viewBox="0 0 8 8"
					>
						<path d="M 0 1 L 7 4 L 0 7 z" fill="#0284c7" />
					</marker>
					<marker
						id="canvas-arrow-amber"
						markerHeight="8"
						markerWidth="8"
						orient="auto-start-reverse"
						refX="7"
						refY="4"
						viewBox="0 0 8 8"
					>
						<path d="M 0 1 L 7 4 L 0 7 z" fill="#d97706" />
					</marker>
					<marker
						id="canvas-arrow-purple"
						markerHeight="8"
						markerWidth="8"
						orient="auto-start-reverse"
						refX="7"
						refY="4"
						viewBox="0 0 8 8"
					>
						<path d="M 0 1 L 7 4 L 0 7 z" fill="#9333ea" />
					</marker>
					<marker
						id="canvas-arrow-emerald"
						markerHeight="8"
						markerWidth="8"
						orient="auto-start-reverse"
						refX="7"
						refY="4"
						viewBox="0 0 8 8"
					>
						<path d="M 0 1 L 7 4 L 0 7 z" fill="#059669" />
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
						{@const relCfg = RELATION_CONFIG[edge.relType || 'default']}
						<g class="group/edge pointer-events-auto">
							<path
								class="{relCfg.stroke} group-hover/edge:opacity-100 transition-colors"
								d={pathD}
								fill="none"
								marker-end={relCfg.marker}
								stroke-dasharray={edge.isAuto ? '6 4' : (relCfg.dash || undefined)}
								stroke-width="2.2"
							/>
							{#if edge.label || !edge.isAuto}
								<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
								<g
									class={!edge.isAuto ? 'cursor-pointer select-none' : 'pointer-events-none select-none'}
									onclick={() => !edge.isAuto && cycleOrSetEdgeRelation(edge.id)}
									onkeydown={(e) => !edge.isAuto && e.key === 'Enter' && cycleOrSetEdgeRelation(edge.id)}
									role={!edge.isAuto ? 'button' : undefined}
									tabindex={!edge.isAuto ? 0 : undefined}
									transform="translate({midX}, {midY})"
								>
									{#if !edge.isAuto}
										<title>点击切换连线关系类型（前置知识 / 因果 / 相似 / 分支追问 / 关联）</title>
									{/if}
									<rect
										class="{relCfg.bg} {relCfg.border} transition-colors"
										height="22"
										rx="6"
										stroke-width="1.2"
										width="64"
										x="-32"
										y="-11"
									/>
									<text
										class="{relCfg.color} text-[10px] font-semibold"
										dominant-baseline="middle"
										text-anchor="middle"
									>
										{edge.label || relCfg.label}
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
									transform="translate({midX + 40}, {midY})"
								>
									<title>删除此连线</title>
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

			<!-- 1. Custom Cards (Hub, Sticky Notes, Image Gallery Cards, RAG Search Cards, SM-2 Flashcards, Knowledge Cards) -->
			{#each customNodes as node (node.id)}
				{@const style = CARD_COLOR_STYLES[node.color]}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-node-id={node.id}
					data-node-kind={node.kind}
					data-knowledge-point-id={node.knowledgePointId ?? ''}
					class="canvas-node-card pointer-events-auto absolute flex flex-col rounded-2xl border bg-background/95 shadow-md backdrop-blur-md transition-shadow hover:shadow-lg {style.border}"
					class:canvas-spotlight-active={provenanceNavigation.spotlightNodeId === node.id}
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
							{#if node.kind === 'knowledge'}
								<button
									type="button"
									class="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary hover:bg-primary/20 transition-colors cursor-pointer"
									onclick={() => (node.historyExpanded = !node.historyExpanded)}
									title="查看版本历史快照与一键回退"
								>
									<span>v{node.currentVersion ?? 1}</span>
									<span class="text-primary/70">· 历史</span>
								</button>
							{/if}

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
									onclick={() => injectIntoChat('请帮我总结当前知识库中的核心概念，并列出结构化大纲')}
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
									<span>拉取待重放</span>
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
						{:else if node.kind === 'knowledge'}
							{@const provList = node.provenance ?? []}
							{@const activeProvIdx = Math.min(node.activeProvenanceIndex ?? 0, Math.max(0, provList.length - 1))}
							{@const currentAnchor = provList[activeProvIdx]}

							<!-- ZONE 1: Immutable Quoted Excerpt Zone -->
							<div class="relative flex flex-col gap-2 rounded-xl border border-primary/25 bg-muted/30 p-3 shadow-inner">
								<!-- Multiple provenance tabs if more than 1 -->
								{#if provList.length > 1}
									<div class="flex flex-wrap items-center gap-1 border-b border-border/40 pb-1.5 mb-0.5">
										<span class="text-[10px] text-muted-foreground mr-1">出处片段:</span>
										{#each provList as prov, pIdx (prov.id || pIdx)}
											<button
												type="button"
												class="rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors {activeProvIdx === pIdx ? 'bg-primary text-primary-foreground font-semibold' : 'bg-muted text-muted-foreground hover:bg-accent'}"
												onclick={() => { node.activeProvenanceIndex = pIdx; persistCanvasState(); }}
											>
												引用 {pIdx + 1}{prov.turnIndex ? ` (#${prov.turnIndex})` : ''}
											</button>
										{/each}
									</div>
								{/if}

								<!-- Provenance Badges Row -->
								<div class="flex flex-wrap items-center gap-1.5">
									{#if currentAnchor?.turnIndex || currentAnchor?.messageId}
										<button
											type="button"
											class="inline-flex items-center gap-1 rounded-md border border-sky-500/30 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-700 dark:text-sky-300 hover:bg-sky-500/20 transition-colors cursor-pointer"
											onclick={() => {
												if (currentAnchor.messageId) {
													void provenanceNavigation.scrollToChatMessage(currentAnchor.messageId);
												}
											}}
											title="点击跳转并高亮左侧对话消息"
										>
											<MessageSquare class="h-3 w-3 text-sky-500" />
											<span>对话 #{currentAnchor.turnIndex ?? 1}</span>
										</button>
									{/if}

									{#if currentAnchor?.docTitle || currentAnchor?.pageStart}
										<button
											type="button"
											class="inline-flex items-center gap-1 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-300 hover:bg-emerald-500/20 transition-colors cursor-pointer"
											onclick={() => {
												provenanceNavigation.openTextbookPreview({
													docId: currentAnchor.docId,
													namespace: getActiveRagNamespace(),
													docTitle: currentAnchor.docTitle || '知识源文档',
													pageStart: currentAnchor.pageStart || 1,
													pageEnd: currentAnchor.pageEnd,
													quoteSnippet: currentAnchor.quoteSnippet
												});
											}}
											title="点击在内嵌预览中查看文档原文对应物理页"
										>
											<BookOpen class="h-3 w-3 text-emerald-500" />
											<span>
												{currentAnchor.docTitle || '文档'} P.{currentAnchor.pageStart ?? 1}{currentAnchor.pageEnd && currentAnchor.pageEnd !== currentAnchor.pageStart ? `-${currentAnchor.pageEnd}` : ''}
											</span>
										</button>
									{/if}
								</div>

								<!-- Immutable Blockquote Quote Text -->
								<blockquote
									class="border-l-3 border-primary/60 bg-primary/5 py-1.5 px-2.5 rounded-r-md font-serif italic text-xs leading-relaxed text-foreground/90 select-text break-words"
								>
									“{currentAnchor?.quoteSnippet || node.rawQuote || node.content}”
								</blockquote>
							</div>

							<!-- ZONE 2: Editable Analysis & Study Zone -->
							<div class="flex flex-col gap-2 pt-1">
								<div class="space-y-1">
									<span class="text-[10px] font-medium text-muted-foreground block">概念标题</span>
									<input
										bind:value={node.title}
										class="w-full rounded-lg border border-border/60 bg-background/80 px-2.5 py-1 text-xs font-semibold text-foreground focus:border-primary focus:outline-hidden"
										onblur={() => handleKnowledgeNodeBlur(node)}
										placeholder="输入概念名称…"
									/>
								</div>

								<div class="space-y-1">
									<div class="flex items-center justify-between">
										<span class="text-[10px] font-medium text-muted-foreground">节点批注与结构化解析 (Markdown)</span>
										<span class="text-[10px] text-muted-foreground/60">失焦自动生成新版本</span>
									</div>
									<textarea
										bind:value={node.content}
										rows="3"
										class="w-full resize-y rounded-lg border border-border/60 bg-background/60 p-2 text-xs leading-relaxed text-foreground placeholder:text-muted-foreground/50 focus:border-primary/50 focus:outline-hidden"
										onblur={() => handleKnowledgeNodeBlur(node)}
										placeholder="在此记录推导过程、关键约束或补充示例…"
									></textarea>
								</div>

								<!-- Difficulty & Mastery Bar -->
								<div class="flex items-center justify-between text-[11px] pt-1.5 border-t border-border/40">
									<div class="flex items-center gap-1">
										<span class="text-[10px] text-muted-foreground">复杂度:</span>
										{#each [{ label: '低', val: 0.3 }, { label: '中', val: 0.5 }, { label: '高', val: 0.8 }] as d}
											<button
												type="button"
												class="rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors cursor-pointer {(node.difficulty ?? 0.5) === d.val ? 'bg-primary text-primary-foreground font-semibold' : 'bg-muted text-muted-foreground hover:bg-accent'}"
												onclick={() => {
													node.difficulty = d.val;
													persistCanvasState();
												}}
											>
												{d.label} {d.val}
											</button>
										{/each}
									</div>

									<div class="flex items-center gap-1.5">
										<span class="text-[10px] text-muted-foreground">稳健度: {Math.round((node.mastery ?? 0.3) * 100)}%</span>
										<div class="h-1.5 w-12 rounded-full bg-muted overflow-hidden">
											<div
												class="h-full bg-emerald-500 transition-all duration-300"
												style="width: {Math.round((node.mastery ?? 0.3) * 100)}%"
											></div>
										</div>
									</div>
								</div>

								<!-- Bottom Action Buttons -->
								<div class="flex flex-wrap items-center justify-between gap-1 pt-1.5 border-t border-border/40">
									<Button
										class="h-6 gap-1 px-2 text-[11px]"
										onclick={() => handleKnowledgeBranchFollowup(node)}
										size="sm"
										variant="ghost"
										title="以此知识点锚定原对话分支并展开深度追问（原分支完整保留）"
									>
										<Sparkles class="h-3 w-3 text-emerald-500" />
										<span>分支追问</span>
									</Button>

									<Button
										class="h-6 gap-1 px-2 text-[11px]"
										onclick={() => convertKnowledgeToFlashcard(node)}
										size="sm"
										variant="outline"
										title="将此知识点转化为记忆闪卡"
									>
										<Brain class="h-3 w-3 text-violet-500" />
										<span>转为重放闪卡</span>
									</Button>

									<Button
										class="h-6 gap-1 px-2 text-[11px]"
										disabled={savingNodeId === node.id}
										onclick={() => {
											void persistKnowledgeNode(node, 'manual_sync').then((ok) => {
												toast[ok ? 'success' : 'warning'](
													ok
														? `已同步知识节点「${node.title}」至后端`
														: '后端同步失败，已本地暂存（下次编辑/加载时重试）'
												);
											});
										}}
										size="sm"
										variant="ghost"
										title="同步至后端知识库"
									>
										{#if savingNodeId === node.id}
											<Loader2 class="h-3 w-3 animate-spin" />
										{:else}
											<Save class="h-3 w-3 text-primary" />
										{/if}
										<span>同步保存</span>
									</Button>
								</div>

								<!-- Revision History Drawer -->
								{#if node.historyExpanded}
									<div class="mt-2 rounded-xl border border-primary/20 bg-background/90 p-2.5 shadow-md space-y-2">
										<div class="flex items-center justify-between border-b border-border/50 pb-1">
											<span class="text-[11px] font-semibold text-foreground flex items-center gap-1">
												<span>📜 版本历史快照</span>
												<span class="text-[10px] text-muted-foreground font-normal">(共 {(node.revisions ?? []).length} 版)</span>
											</span>
											<button
												type="button"
												class="text-[10px] text-muted-foreground hover:text-foreground cursor-pointer"
												onclick={() => (node.historyExpanded = false)}
											>
												收起 ✕
											</button>
										</div>

										<div class="max-h-48 overflow-y-auto space-y-2 pr-1">
											{#each (node.revisions ?? []).slice().reverse() as rev (rev.version)}
												<div class="rounded-lg border border-border/60 bg-muted/30 p-2 text-[11px]">
													<div class="flex items-center justify-between mb-1">
														<div class="flex items-center gap-1.5">
															<span class="font-mono font-bold text-primary">v{rev.version}</span>
															<span class="text-[10px] text-muted-foreground">
																{new Date(rev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
															</span>
															<span class="rounded bg-primary/10 px-1 py-0.2 text-[9px] font-medium text-primary">
																{rev.changeType}
															</span>
														</div>
														{#if rev.version !== (node.currentVersion ?? 1)}
															<button
																type="button"
																class="rounded bg-primary/15 hover:bg-primary/25 px-1.5 py-0.5 text-[10px] font-medium text-primary transition-colors cursor-pointer"
																onclick={() => handleKnowledgeRollback(node, rev)}
															>
																回退至此版本
															</button>
														{:else}
															<span class="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">当前版本</span>
														{/if}
													</div>
													<div class="text-[10px] text-muted-foreground mb-1">
														{rev.summary}
													</div>
													<div class="line-clamp-2 text-[11px] text-foreground/80 font-mono bg-background/50 rounded p-1">
														{rev.content || '空内容'}
													</div>
												</div>
											{/each}
										</div>
									</div>
								{/if}
							</div>
						{/if}
					</div>
				</div>
			{/each}

			<!-- 2. Automatic Conversation Turn Nodes (Top Image Carousel Dock + User Prompt + Linked Knowledge/Memory + Assistant Response) -->
			{#each chatTurnNodes as turn (turn.id)}
				{@const reasoningText = sanitizeUtf8Text(
					turn.assistantMessage?.reasoningContent || turn.assistantMessage?.thinking || ''
				)}
				{@const isStreamingTurn =
					turn === chatTurnNodes[chatTurnNodes.length - 1] &&
					(chatStore.isStreaming() || chatStore.isLoading)}
				{@const assistantText =
					turn.assistantMessage.content ||
					(isStreamingTurn ? '正在生成回复…' : '（本轮生成已取消）')}
				{@const isReasonOpen = expandedReasoning[turn.id] ?? false}
				{@const assoc = turnAssociations[turn.id]}
				{@const turnImages = getTurnImages(turn)}
				{@const turnDocs = getTurnDocs(turn)}
				{@const connectedNodes = getConnectedCustomNodes(turn.id)}
				{@const placement = assoc?.imagePlacement ?? 'top'}
				{@const branchInfo = getTurnBranchInfo(turn)}
				{@const activeImgIdx = Math.min(
					assoc?.activeImageIndex ?? 0,
					Math.max(0, turnImages.length - 1)
				)}
				{@const currentTurnImg = turnImages[activeImgIdx]}

				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-node-id={turn.id}
					data-turn-index={turn.turnIndex}
					data-message-id={turn.assistantMessage?.id ?? turn.userMessage?.id}
					data-message-role="assistant"
					class="canvas-node-card pointer-events-auto absolute flex flex-col rounded-2xl border border-border/80 bg-background/95 shadow-md backdrop-blur-md transition-all hover:border-primary/50 hover:shadow-xl"
					class:canvas-spotlight-active={provenanceNavigation.spotlightNodeId === turn.id}
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
							{#if branchInfo}
								<span
									class="inline-flex shrink-0 items-center gap-0.5 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-1 py-0.5 text-[10px] font-medium text-emerald-700 dark:text-emerald-300"
									title="该消息存在多个对话分支，点击切换（原分支完整保留）"
								>
									<button
										aria-label="切换到上一个对话分支"
										class="rounded px-0.5 hover:bg-emerald-500/20"
										onclick={(e) => {
											e.stopPropagation();
											void switchTurnBranch(turn, -1);
										}}
										type="button"
									>
										<ChevronLeft class="h-3 w-3" />
									</button>
									<span>分支 {branchInfo.current}/{branchInfo.total}</span>
									<button
										aria-label="切换到下一个对话分支"
										class="rounded px-0.5 hover:bg-emerald-500/20"
										onclick={(e) => {
											e.stopPropagation();
											void switchTurnBranch(turn, 1);
										}}
										type="button"
									>
										<ChevronRight class="h-3 w-3" />
									</button>
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
								title="上传参考图片或展开卡片内的配图展台"
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
								class="rounded-md px-1.5 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
								onclick={() => handleTurnBranchFollowup(turn)}
								title="围绕此轮对话锚定历史分支并展开追问（原分支完整保留）"
								type="button"
							>
								🌿 分支追问
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
										<span>关联参考图片 ({activeImgIdx + 1}/{turnImages.length})</span>
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
											🔍 点击放大查看
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
											<span>关联会话图片 ({allConversationImages.length}张)</span>
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
													{mem.flipped ? '🔓 背面解析' : '❓ 正面要点'}
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
										content={assistantText}
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

<style>
	:global(.canvas-world-layer--navigating) {
		transition: transform 450ms cubic-bezier(0.16, 1, 0.3, 1) !important;
	}

	:global(.canvas-spotlight-active) {
		animation: canvas-card-spotlight 2.5s cubic-bezier(0.25, 1, 0.5, 1) forwards !important;
		z-index: 45 !important;
	}

	@keyframes canvas-card-spotlight {
		0% {
			box-shadow: 0 0 0 0 var(--color-primary, #6366f1), 0 0 25px 2px color-mix(in oklab, var(--color-primary, #6366f1) 50%, transparent);
			transform: scale(1);
		}
		25% {
			box-shadow: 0 0 0 8px color-mix(in oklab, var(--color-primary, #6366f1) 30%, transparent), 0 0 45px 8px color-mix(in oklab, var(--color-primary, #6366f1) 60%, transparent);
			transform: scale(1.025);
		}
		60% {
			box-shadow: 0 0 0 14px transparent, 0 0 20px 2px color-mix(in oklab, var(--color-primary, #6366f1) 25%, transparent);
			transform: scale(1.01);
		}
		100% {
			box-shadow: 0 0 0 0 transparent;
			transform: scale(1);
		}
	}
</style>
