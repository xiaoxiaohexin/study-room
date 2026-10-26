<script lang="ts">
	import {
		AlertCircle,
		BookOpen,
		CheckCircle2,
		Database,
		FileText,
		FileUp,
		Globe,
		HardDrive,
		Inbox,
		Layers,
		Loader2,
		RefreshCw,
		Search,
		Sparkles,
		Trash2,
		UploadCloud
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { conversationsStore, deviceStore } from '$lib/stores';
	import { getActiveCourseId, getActiveRagNamespace, learningHeaders } from '$lib/services/learning-context';
	import {
		deleteStagingDocument,
		distillStagingDocument,
		getWebStatus,
		listStagingCandidates,
		listStagingDocuments,
		previewUrl,
		promoteStaging,
		rejectStagingCandidate,
		revokePromotion,
		searchWeb,
		stageUrl,
		type StagingCandidate,
		type StagingDocument,
		type WebPreviewResult,
		type WebStatus
	} from '$lib/services/web-staging.service';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		open?: boolean;
		onOpenChange?: (open: boolean) => void;
	}

	let { onOpenChange, open = $bindable(false) }: Props = $props();

	interface IngestedDoc {
		doc_id: string;
		filename: string;
		chunks: number;
		created_at: number;
		namespace: string;
	}

	interface RagStats {
		namespace: string;
		collection: string;
		total_chunks: number;
		embedder: string;
		embedder_dim: number;
		vector_store_status: string;
	}

	interface SearchResultItem {
		content: string;
		score: number;
		source?: string;
		filename?: string;
		headers?: string[];
	}

	let activeTab = $state<'list' | 'upload' | 'test' | 'web' | 'staging'>('upload');
	// Default to the knowledge base bound to the active conversation so uploading
	// and browsing do not silently target a different namespace.
	let namespace = $state(getActiveRagNamespace());
	let stats = $state<RagStats | null>(null);
	let documents = $state<IngestedDoc[]>([]);
	let loadingDocs = $state(false);

	// Upload states
	let selectedFile = $state<File | null>(null);
	let uploading = $state(false);
	let uploadSuccessMessage = $state<string | null>(null);
	let uploadErrorMessage = $state<string | null>(null);
	let fileInputRef = $state<HTMLInputElement | null>(null);

	// Test search states
	let testQuery = $state('MAC地址的标准长度是多少？');
	let searching = $state(false);
	let searchResults = $state<SearchResultItem[]>([]);
	let searchPerformed = $state(false);

	// Deleting doc state
	let deletingDocId = $state<string | null>(null);

	// 联网导入 / 临时知识库状态
	let webStatus = $state<WebStatus | null>(null);
	let webUrl = $state('');
	let webQuery = $state('');
	let webPreview = $state<WebPreviewResult | null>(null);
	let webBusy = $state(false);
	let stagingDocuments = $state<StagingDocument[]>([]);
	let stagingCandidates = $state<StagingCandidate[]>([]);
	let stagingBusy = $state(false);
	let promoting = $state(false);
	let stagingCourseId = $state(getActiveCourseId());
	let selectedCandidateIds = $state<string[]>([]);

	function stagingSessionId(): string {
		return conversationsStore.activeConversation?.id ?? 'manual_import';
	}

	function handleOpenChange(value: boolean) {
		open = value;
		onOpenChange?.(value);
		if (value) {
			namespace = getActiveRagNamespace();
			stagingCourseId = getActiveCourseId();
			void loadAllData();
			void loadWebStatus();
			void loadStagingData();
		}
	}

	async function loadWebStatus() {
		try {
			webStatus = await getWebStatus();
		} catch (err) {
			console.warn('Failed to load web status:', err);
			webStatus = null;
		}
	}

	async function loadStagingData() {
		stagingBusy = true;
		try {
			const [docs, candidates] = await Promise.all([
				listStagingDocuments({ limit: 100 }),
				listStagingCandidates({ status: 'pending', limit: 100 })
			]);
			stagingDocuments = docs.documents;
			stagingCandidates = candidates.candidates;
		} catch (err) {
			console.warn('Failed to load staging data:', err);
		} finally {
			stagingBusy = false;
		}
	}

	async function handleWebPreview() {
		const url = webUrl.trim();
		if (!url) {
			toast.error('请输入网页地址');
			return;
		}
		webBusy = true;
		webPreview = null;
		try {
			webPreview = await previewUrl(url);
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			webBusy = false;
		}
	}

	async function handleWebStage() {
		if (!webPreview) return;
		webBusy = true;
		try {
			const result = await stageUrl(webPreview.final_url || webPreview.url, {
				sessionId: stagingSessionId(),
				conversationId: conversationsStore.activeConversation?.id ?? '',
				query: webQuery.trim(),
				targetCourseId: stagingCourseId
			});
			toast.success(
				result.document.reused
					? '该网页内容已在临时知识库中（自动去重）'
					: '已存入临时知识库，等待提纯晋升'
			);
			webPreview = null;
			webUrl = '';
			await loadStagingData();
			activeTab = 'staging';
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			webBusy = false;
		}
	}

	async function handleWebSearchStage() {
		const query = webQuery.trim();
		if (!query) {
			toast.error('请输入联网搜索关键词');
			return;
		}
		webBusy = true;
		try {
			const result = await searchWeb(query, {
				topK: 3,
				stage: true,
				sessionId: stagingSessionId()
			});
			const count = result.staged?.length ?? 0;
			const provider = result.results?.[0]?.provider;
			const source = provider ? `（搜索源：${provider}）` : '';
			toast.success(
				count > 0
					? `搜索完成${source}，已抓取 ${count} 篇网页存入临时知识库`
					: `搜索完成${source}，但未成功抓取到网页`
			);
			await loadStagingData();
			if (count > 0) activeTab = 'staging';
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			webBusy = false;
		}
	}

	async function handleDistill(doc: StagingDocument) {
		stagingBusy = true;
		try {
			const result = await distillStagingDocument(doc.doc_id);
			toast.success(`已提纯出 ${result.candidates.length} 个知识候选（质量分 ${(result.quality_score * 100).toFixed(0)}）`);
			await loadStagingData();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			stagingBusy = false;
		}
	}

	async function handleDeleteStaging(doc: StagingDocument) {
		if (!confirm(`确定丢弃临时文档「${doc.title}」吗？`)) return;
		stagingBusy = true;
		try {
			await deleteStagingDocument(doc.doc_id);
			toast.success('已丢弃临时文档');
			await loadStagingData();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			stagingBusy = false;
		}
	}

	function toggleCandidate(id: string) {
		selectedCandidateIds = selectedCandidateIds.includes(id)
			? selectedCandidateIds.filter((item) => item !== id)
			: [...selectedCandidateIds, id];
	}

	async function handlePromote() {
		const selected = stagingCandidates.filter((c) => selectedCandidateIds.includes(c.id));
		const docIds = Array.from(new Set(selected.map((c) => c.doc_id)));
		if (selected.length === 0) {
			toast.error('请先勾选要晋升的提纯候选');
			return;
		}
		promoting = true;
		try {
			const result = await promoteStaging({
				stagingDocIds: docIds,
				candidateIds: selectedCandidateIds,
				targetCourseId: stagingCourseId.trim() || 'default'
			});
			if (result.ok) {
				toast.success(
					`已晋升 ${result.promoted_documents.length} 篇文档 / ${result.promoted_knowledge_points.length} 个知识节点到「${result.target_namespace}」`
				);
			} else {
				toast.warning(`部分晋升失败：${result.errors.join('；')}`);
			}
			selectedCandidateIds = [];
			await loadStagingData();
			await loadAllData();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			promoting = false;
		}
	}

	async function handleRejectCandidate(candidate: StagingCandidate) {
		try {
			await rejectStagingCandidate(candidate.id);
			toast.success('已驳回该候选');
			await loadStagingData();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		}
	}

	async function handleRevoke(doc: StagingDocument) {
		if (!confirm(`确定撤销「${doc.title}」的晋升吗？对应标准文档与知识节点会被删除。`)) return;
		stagingBusy = true;
		try {
			const result = await revokePromotion(doc.doc_id);
			toast.success(
				`已撤销晋升（删除文档 ${result.removed_doc_id || '无'}，知识节点 ${result.revoked_knowledge_points.length} 个）`
			);
			await loadStagingData();
			await loadAllData();
		} catch (err) {
			toast.error(err instanceof Error ? err.message : String(err));
		} finally {
			stagingBusy = false;
		}
	}

	async function loadAllData() {
		loadingDocs = true;
		try {
			const authHeaders = learningHeaders();
			const [statsRes, docsRes] = await Promise.all([
				fetch(`/api/rag/stats?namespace=${encodeURIComponent(namespace)}`, {
					headers: authHeaders
				}),
				fetch(`/api/rag/documents?namespace=${encodeURIComponent(namespace)}`, {
					headers: authHeaders
				})
			]);

			if (statsRes.ok) {
				stats = (await statsRes.json()) as RagStats;
			}
			if (docsRes.ok) {
				const docsJson = await docsRes.json();
				documents = (docsJson.documents ?? []) as IngestedDoc[];
			}
		} catch (err) {
			console.error('Failed to load RAG data:', err);
		} finally {
			loadingDocs = false;
		}
	}

	function handleFileChange(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			selectedFile = target.files[0];
			uploadSuccessMessage = null;
			uploadErrorMessage = null;
		}
	}

	function handleDrop(event: DragEvent) {
		event.preventDefault();
		if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
			selectedFile = event.dataTransfer.files[0];
			uploadSuccessMessage = null;
			uploadErrorMessage = null;
		}
	}

	async function handleUpload() {
		if (!selectedFile) {
			toast.error('请先选择要导入的文件');
			return;
		}

		uploading = true;
		uploadSuccessMessage = null;
		uploadErrorMessage = null;

		try {
			const formData = new FormData();
			formData.append('file', selectedFile);
			formData.append('namespace', namespace);

			const resp = await fetch('/api/rag/ingest', {
				body: formData,
				headers: learningHeaders(),
				method: 'POST'
			});

			if (!resp.ok) {
				const errText = await resp.text();
				throw new Error(errText || '上传切块失败');
			}

			const result = (await resp.json()) as {
				doc_id: string;
				chunks: number;
				filename: string;
				namespace: string;
			};

			uploadSuccessMessage = `✅ 文件「${result.filename}」装载成功！已完成多格式解析、Markdown 结构切块并提取出 ${result.chunks} 个语义向量片段（文档 ID: ${result.doc_id}）。`;
			toast.success(`成功导入知识库文档：${result.filename}（${result.chunks} 块）`);
			selectedFile = null;
			if (fileInputRef) fileInputRef.value = '';

			await loadAllData();
			activeTab = 'list';
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			uploadErrorMessage = `导入失败：${msg}`;
			toast.error(`知识库导入失败：${msg}`);
		} finally {
			uploading = false;
		}
	}

	async function handleDeleteDoc(doc: IngestedDoc) {
		if (!confirm(`确定要从知识库中移除文档「${doc.filename}」及其全部切块吗？`)) {
			return;
		}

		deletingDocId = doc.doc_id;
		try {
			const resp = await fetch(
				`/api/rag/documents/${encodeURIComponent(doc.doc_id)}?namespace=${encodeURIComponent(namespace)}`,
				{ method: 'DELETE', headers: learningHeaders() }
			);

			if (!resp.ok) {
				throw new Error('删除文档失败');
			}

			toast.success(`已从知识库删除文档：${doc.filename}`);
			await loadAllData();
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			toast.error(`删除失败：${msg}`);
		} finally {
			deletingDocId = null;
		}
	}

	async function handleRunTestSearch() {
		const q = testQuery.trim();
		if (!q) {
			toast.error('请输入检索问题或测试词');
			return;
		}

		searching = true;
		searchPerformed = true;
		searchResults = [];

		try {
			const resp = await fetch('/api/rag/search', {
				body: JSON.stringify({
					expand: true,
					namespace,
					query: q,
					top_k: 5
				}),
				headers: {
					'Content-Type': 'application/json',
					...learningHeaders()
				},
				method: 'POST'
			});

			if (!resp.ok) {
				throw new Error('知识库检索请求失败');
			}

			const data = await resp.json();
			const list = (data.results ?? []) as Array<{
				content?: string;
				score?: number;
				payload?: { content?: string; filename?: string; headers?: string[] };
				source?: string;
			}>;

			searchResults = list.map((item) => ({
				content: item.payload?.content ?? item.content ?? '',
				filename: item.payload?.filename ?? item.source ?? '未知文件',
				headers: item.payload?.headers,
				score: typeof item.score === 'number' ? item.score : 0.85
			}));

			if (searchResults.length > 0) {
				toast.success(`检索完成，命中 ${searchResults.length} 条高相关片段`);
			} else {
				toast.info('未检索到匹配的切块，可尝试调整关键词或导入更多知识文档');
			}
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			toast.error(`检索异常：${msg}`);
		} finally {
			searching = false;
		}
	}

	function formatBytes(bytes: number) {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
	}

	function formatTime(timestamp: number) {
		if (!timestamp) return '刚刚';
		const date = new Date(timestamp * 1000);
		return date.toLocaleString('zh-CN', {
			day: '2-digit',
			hour: '2-digit',
			minute: '2-digit',
			month: '2-digit'
		});
	}

	$effect(() => {
		if (open) {
			void loadAllData();
			void loadWebStatus();
			void loadStagingData();
		}
	});
</script>

<Dialog.Root onOpenChange={handleOpenChange} {open}>
	<Dialog.Content
		class="flex flex-col p-5 max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-5rem)]! md:max-h-[760px]! md:w-[calc(100vw-5rem)]! md:max-w-[880px]! overflow-hidden"
		onOpenAutoFocus={(event) => deviceStore.isMobile && event.preventDefault()}
	>
		<!-- Dialog Header with dedicated pr-14 padding to prevent collision with ShadCN's absolute top-4 right-4 close button -->
		<Dialog.Header class="relative flex flex-row items-center justify-between border-b border-border/60 pb-3.5 pr-14 shrink-0">
			<div class="flex items-center gap-3 min-w-0 pr-3">
				<span class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary/20 bg-primary/10 text-primary shadow-xs">
					<BookOpen class="h-5 w-5" />
				</span>
				<div class="min-w-0">
					<div class="flex flex-wrap items-center gap-2">
						<Dialog.Title class="text-base font-semibold text-foreground">
							知识库管理与装载 (RAG)
						</Dialog.Title>
						<span class="rounded-md border border-border/80 bg-muted/60 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
							空间: {namespace}
						</span>
					</div>
					<Dialog.Description class="mt-0.5 text-[12px] text-muted-foreground font-normal leading-relaxed">
						支持上传专业资料、PDF 文档、技术手册与笔记，自动解析结构切块并存入本地持久化向量库供智能体检索调用
					</Dialog.Description>
				</div>
			</div>

			<Button
				class="h-8 shrink-0 gap-1.5 px-3 text-xs text-muted-foreground hover:bg-muted hover:text-foreground border border-border/80 shadow-xs transition-colors"
				disabled={loadingDocs}
				onclick={loadAllData}
				size="sm"
				variant="outline"
			>
				<RefreshCw class="h-3.5 w-3.5 {loadingDocs ? 'animate-spin' : ''}" />
				<span>刷新状态</span>
			</Button>
		</Dialog.Header>

		<!-- Sub-header Stats Dashboard -->
		<div class="my-3 grid grid-cols-2 gap-2 sm:grid-cols-4 shrink-0 text-xs">
			<div class="flex items-center gap-2.5 rounded-xl border border-border/70 bg-card/60 p-2.5 shadow-xs transition-colors hover:bg-muted/40">
				<div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
					<Database class="h-4 w-4" />
				</div>
				<div class="min-w-0">
					<div class="text-[10px] text-muted-foreground font-medium">向量库状态</div>
					<div class="truncate font-semibold text-foreground text-[11px]">
						{stats?.vector_store_status === 'ok:sqlite_vector'
							? 'SQLite持久化'
							: stats?.vector_store_status ?? '正常运行'}
					</div>
				</div>
			</div>

			<div class="flex items-center gap-2.5 rounded-xl border border-border/70 bg-card/60 p-2.5 shadow-xs transition-colors hover:bg-muted/40">
				<div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-sky-500/10 text-sky-500">
					<Layers class="h-4 w-4" />
				</div>
				<div class="min-w-0">
					<div class="text-[10px] text-muted-foreground font-medium">嵌入模型 (Embedder)</div>
					<div class="truncate font-semibold text-foreground text-[11px]" title="{stats?.embedder ?? 'bge-small-zh'} (512维)">
						{stats?.embedder === 'emb512' ? 'bge-small-zh' : stats?.embedder ?? 'bge-small-zh'} (512维)
					</div>
				</div>
			</div>

			<div class="flex items-center gap-2.5 rounded-xl border border-border/70 bg-card/60 p-2.5 shadow-xs transition-colors hover:bg-muted/40">
				<div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500">
					<FileText class="h-4 w-4" />
				</div>
				<div class="min-w-0">
					<div class="text-[10px] text-muted-foreground font-medium">收录文档总数</div>
					<div class="font-semibold text-foreground text-[11px]">
						{documents.length} 篇
					</div>
				</div>
			</div>

			<div class="flex items-center gap-2.5 rounded-xl border border-border/70 bg-card/60 p-2.5 shadow-xs transition-colors hover:bg-muted/40">
				<div class="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-violet-500">
					<HardDrive class="h-4 w-4" />
				</div>
				<div class="min-w-0">
					<div class="text-[10px] text-muted-foreground font-medium">向量切块索引总数</div>
					<div class="font-semibold text-foreground text-[11px]">
						{stats?.total_chunks ?? 0} 个切块
					</div>
				</div>
			</div>
		</div>

		<!-- Tab Navigation Segmented Bar -->
		<div class="flex shrink-0 items-center gap-1 rounded-xl border border-border/60 bg-muted/40 p-1 mb-2">
			<button
				class="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all {activeTab ===
				'upload'
					? 'bg-background text-foreground shadow-xs font-semibold'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'upload')}
				type="button"
			>
				<UploadCloud class="h-3.5 w-3.5 text-primary" />
				<span>导入新文档 / 资料</span>
			</button>

			<button
				class="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all {activeTab ===
				'list'
					? 'bg-background text-foreground shadow-xs font-semibold'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'list')}
				type="button"
			>
				<FileText class="h-3.5 w-3.5 text-emerald-500" />
				<span>已入库文档列表 ({documents.length})</span>
			</button>

			<button
				class="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all {activeTab ===
				'test'
					? 'bg-background text-foreground shadow-xs font-semibold'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'test')}
				type="button"
			>
				<Search class="h-3.5 w-3.5 text-sky-500" />
				<span>检索测试</span>
			</button>

			<button
				class="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all {activeTab ===
				'web'
					? 'bg-background text-foreground shadow-xs font-semibold'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'web')}
				type="button"
			>
				<Globe class="h-3.5 w-3.5 text-indigo-500" />
				<span>联网导入{webStatus?.enabled ? '' : '（未开启）'}</span>
			</button>

			<button
				class="flex flex-1 items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs font-medium transition-all {activeTab ===
				'staging'
					? 'bg-background text-foreground shadow-xs font-semibold'
					: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => (activeTab = 'staging')}
				type="button"
			>
				<Inbox class="h-3.5 w-3.5 text-amber-500" />
				<span>临时知识库 ({stagingDocuments.length})</span>
			</button>
		</div>

		<!-- Tab Body Container -->
		<div class="min-h-0 flex-1 overflow-y-auto py-3">
			<!-- TAB 1: UPLOAD / INGEST -->
			{#if activeTab === 'upload'}
				<div class="space-y-4">
					<!-- Drag & Drop Zone -->
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						class="flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border/80 bg-muted/20 p-8 text-center transition-colors hover:border-primary/60 hover:bg-muted/40 cursor-pointer"
						ondragover={(e) => e.preventDefault()}
						ondrop={handleDrop}
						onclick={() => fileInputRef?.click()}
						onkeydown={(e) => e.key === 'Enter' && fileInputRef?.click()}
						role="button"
						tabindex="0"
					>
						<input
							accept=".pdf,.txt,.md,.markdown,.docx,.json,.csv,.py,.sql,.jpg,.jpeg,.png,.webp"
							bind:this={fileInputRef}
							class="hidden"
							onchange={handleFileChange}
							type="file"
						/>

						<div class="mb-3 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary">
							<FileUp class="h-7 w-7" />
						</div>

						<div class="text-sm font-semibold text-foreground">
							{selectedFile ? `已选择：${selectedFile.name}` : '点击选择或拖拽文件到此处'}
						</div>
						<p class="mt-1 text-xs text-muted-foreground max-w-md">
							支持 PDF 文档、电子书、Markdown、Word(Docx)、纯文本、代码或图表扫描件。后端将智能通过 MarkItDown 提取内容并自动按标题层级分块。
						</p>

						<div class="mt-3 flex flex-wrap items-center justify-center gap-1.5 text-[10px] text-muted-foreground">
							<span class="rounded bg-muted px-1.5 py-0.5">PDF</span>
							<span class="rounded bg-muted px-1.5 py-0.5">Markdown (.md)</span>
							<span class="rounded bg-muted px-1.5 py-0.5">Word (.docx)</span>
							<span class="rounded bg-muted px-1.5 py-0.5">TXT / JSON</span>
							<span class="rounded bg-muted px-1.5 py-0.5">代码 (.py/.sql)</span>
							<span class="rounded bg-muted px-1.5 py-0.5">图表扫描件 OCR</span>
						</div>

						{#if selectedFile}
							<div class="mt-3 flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/10 px-3 py-1 text-xs text-primary font-medium">
								<FileText class="h-3.5 w-3.5" />
								<span>{selectedFile.name} ({formatBytes(selectedFile.size)})</span>
							</div>
						{/if}
					</div>

					<!-- Upload Actions Bar -->
					<div class="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/70 bg-muted/30 p-3">
						<div class="flex items-center gap-2 text-xs">
							<span class="text-muted-foreground font-medium">指定知识库命名空间:</span>
							<Input
								bind:value={namespace}
								class="h-7 w-32 text-xs"
								placeholder="如 default / math"
							/>
						</div>

						<Button
							class="gap-1.5 text-xs font-semibold"
							disabled={!selectedFile || uploading}
							onclick={handleUpload}
						>
							{#if uploading}
								<Loader2 class="h-3.5 w-3.5 animate-spin" />
								<span>正在结构解析、切块与向量嵌入…</span>
							{:else}
								<UploadCloud class="h-3.5 w-3.5" />
								<span>开始导入并向量化入库</span>
							{/if}
						</Button>
					</div>

					<!-- Feedback Messages -->
					{#if uploadSuccessMessage}
						<div class="flex items-start gap-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3.5 text-xs text-emerald-700 dark:text-emerald-300">
							<CheckCircle2 class="h-4 w-4 shrink-0 mt-0.5" />
							<div>
								<div class="font-semibold">导入成功</div>
								<p class="mt-0.5 leading-relaxed">{uploadSuccessMessage}</p>
							</div>
						</div>
					{/if}

					{#if uploadErrorMessage}
						<div class="flex items-start gap-2.5 rounded-xl border border-destructive/30 bg-destructive/10 p-3.5 text-xs text-destructive">
							<AlertCircle class="h-4 w-4 shrink-0 mt-0.5" />
							<div>
								<div class="font-semibold">导入遇到错误</div>
								<p class="mt-0.5 leading-relaxed">{uploadErrorMessage}</p>
							</div>
						</div>
					{/if}
				</div>

			<!-- TAB 2: DOCUMENT LIST -->
			{:else if activeTab === 'list'}
				<div class="space-y-2">
					{#if loadingDocs}
						<div class="flex flex-col items-center justify-center py-12 text-muted-foreground">
							<Loader2 class="h-6 w-6 animate-spin text-primary" />
							<p class="mt-2 text-xs">正在读取知识库文档列表…</p>
						</div>
					{:else if documents.length === 0}
						<div class="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border py-12 text-center">
							<BookOpen class="h-8 w-8 text-muted-foreground/50 mb-2" />
							<p class="text-sm font-medium text-foreground">当前空间暂无已收录知识库文档</p>
							<p class="mt-1 text-xs text-muted-foreground">点击上方「导入新文档」将参考文档录入向量库</p>
							<Button
								class="mt-4 gap-1 text-xs"
								onclick={() => (activeTab = 'upload')}
								size="sm"
							>
								<UploadCloud class="h-3.5 w-3.5" />
								<span>立刻导入第一篇文档</span>
							</Button>
						</div>
					{:else}
						<div class="space-y-2">
							{#each documents as doc (doc.doc_id)}
								<div class="flex items-center justify-between gap-3 rounded-xl border border-border/70 bg-card p-3 shadow-xs transition-colors hover:border-primary/40">
									<div class="flex items-center gap-3 min-w-0">
										<span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
											<FileText class="h-4 w-4" />
										</span>
										<div class="min-w-0">
											<div class="flex items-center gap-2">
												<span class="truncate text-xs font-semibold text-foreground" title={doc.filename}>
													{doc.filename}
												</span>
												<span class="rounded bg-muted px-1.5 py-0.2 font-mono text-[10px] text-muted-foreground">
													ID: {doc.doc_id}
												</span>
											</div>
											<div class="mt-0.5 flex items-center gap-3 text-[11px] text-muted-foreground">
												<span>切块数：<strong class="text-foreground">{doc.chunks}</strong> 个语义片段</span>
												<span>·</span>
												<span>入库时间：{formatTime(doc.created_at)}</span>
												<span>·</span>
												<span class="font-mono">命名空间：{doc.namespace}</span>
											</div>
										</div>
									</div>

									<div class="flex shrink-0 items-center gap-2">
										<Button
											class="h-7 text-xs text-destructive hover:bg-destructive/15"
											disabled={deletingDocId === doc.doc_id}
											onclick={() => handleDeleteDoc(doc)}
											size="sm"
											variant="ghost"
										>
											{#if deletingDocId === doc.doc_id}
												<Loader2 class="h-3.5 w-3.5 animate-spin" />
											{:else}
												<Trash2 class="h-3.5 w-3.5" />
											{/if}
											<span class="ml-1">移除</span>
										</Button>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>

			<!-- TAB 3: SEARCH TEST -->
			{:else if activeTab === 'test'}
				<div class="space-y-3">
					<div class="flex gap-2">
						<Input
							bind:value={testQuery}
							class="text-xs"
							onkeydown={(e) => e.key === 'Enter' && handleRunTestSearch()}
							placeholder="输入要检索测试的问题或核心概念（如：以太网 MAC 地址格式）…"
						/>
						<Button
							class="gap-1.5 shrink-0 text-xs"
							disabled={searching}
							onclick={handleRunTestSearch}
						>
							{#if searching}
								<Loader2 class="h-3.5 w-3.5 animate-spin" />
							{:else}
								<Search class="h-3.5 w-3.5" />
							{/if}
							<span>执行检索</span>
						</Button>
					</div>

					<p class="text-[11px] text-muted-foreground">
						提示：这里使用的是与大模型对话时完全一致的 RAG 语义检索流水线（查询扩充 + 向量余弦比对 + TopK 筛选）。
					</p>

					<!-- Results List -->
					<div class="space-y-2 mt-2">
						{#if searching}
							<div class="flex items-center justify-center py-8 text-xs text-muted-foreground gap-2">
								<Loader2 class="h-4 w-4 animate-spin text-primary" />
								<span>正在进行高维向量空间余弦匹配…</span>
							</div>
						{:else if searchPerformed && searchResults.length === 0}
							<div class="rounded-xl border border-dashed border-border p-6 text-center text-xs text-muted-foreground">
								未检索到相似度高于阈值的片段。请检查是否已在对应命名空间装载相关资料。
							</div>
						{:else if searchResults.length > 0}
							{#each searchResults as item, idx (idx)}
								<div class="rounded-xl border border-border/80 bg-muted/20 p-3 space-y-1.5 text-xs">
									<div class="flex items-center justify-between text-[11px] text-muted-foreground border-b border-border/40 pb-1.5">
										<div class="flex items-center gap-1.5">
											<span class="flex h-5 w-5 items-center justify-center rounded-md bg-primary/10 font-bold text-primary text-[10px]">
												#{idx + 1}
											</span>
											<span class="font-medium text-foreground">{item.filename}</span>
											{#if item.headers && item.headers.length > 0}
												<span class="text-muted-foreground">› {item.headers.join(' › ')}</span>
											{/if}
										</div>
										<span class="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] font-mono font-semibold text-emerald-600 dark:text-emerald-400">
											匹配度 {(item.score * 100).toFixed(1)}%
										</span>
									</div>
									<p class="text-[12px] leading-relaxed text-foreground/90 whitespace-pre-wrap font-sans">
										{item.content}
									</p>
								</div>
							{/each}
						{/if}
					</div>
				</div>
			<!-- TAB 4: WEB IMPORT -->
			{:else if activeTab === 'web'}
				<div class="space-y-3">
					{#if !webStatus?.enabled}
						<div class="flex items-start gap-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-700 dark:text-amber-300">
							<AlertCircle class="mt-0.5 h-4 w-4 shrink-0" />
							<div>
								<p class="font-semibold">联网功能未开启</p>
								<p class="mt-0.5 text-[11px] leading-relaxed">
									在项目根目录 .env 中设置 <code class="rounded bg-muted px-1">WEB_ENABLED=true</code> 并重启后端；
									如需搜索发现，再配置 <code class="rounded bg-muted px-1">WEB_SEARCH_PROVIDER</code> 或
									<code class="rounded bg-muted px-1">MCP_WEB_SERVER_URL</code>。
								</p>
							</div>
						</div>
					{/if}

					{#if webStatus?.enabled}
						<div class="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-border/60 bg-muted/30 px-2.5 py-1.5 text-[11px] text-muted-foreground">
							<Globe class="h-3 w-3 shrink-0 text-indigo-500" />
							<span>
								搜索源：{webStatus.search_providers?.length
									? webStatus.search_providers.join(' → ')
									: '未配置'}
							</span>
							{#if webStatus.search_base_url}
								<span class="font-mono">{webStatus.search_base_url}</span>
							{/if}
							{#if webStatus.mcp_configured}
								<span>· MCP 已配置（需服务在线）</span>
							{/if}
							{#if !webStatus.search_available}
								<span class="text-amber-600 dark:text-amber-400">
									· 搜索不可用（仍可直接粘贴 URL 抓取）
								</span>
							{/if}
						</div>
					{/if}

					<div class="flex gap-2">
						<Input
							bind:value={webUrl}
							class="text-xs"
							onkeydown={(e) => e.key === 'Enter' && handleWebPreview()}
							placeholder="粘贴网页 URL（https://…），抓取正文后存入临时知识库…"
						/>
						<Button
							class="gap-1.5 shrink-0 text-xs"
							disabled={webBusy || !webStatus?.enabled}
							onclick={handleWebPreview}
						>
							{#if webBusy}
								<Loader2 class="h-3.5 w-3.5 animate-spin" />
							{:else}
								<Globe class="h-3.5 w-3.5" />
							{/if}
							<span>抓取预览</span>
						</Button>
					</div>

					{#if webPreview}
						<div class="space-y-2 rounded-xl border border-indigo-500/30 bg-indigo-500/5 p-3">
							<div class="flex flex-wrap items-center justify-between gap-2">
								<div class="min-w-0">
									<p class="truncate text-xs font-semibold text-foreground">{webPreview.title}</p>
									<p class="truncate font-mono text-[10px] text-muted-foreground">
										{webPreview.final_url} · {webPreview.chars} 字
									</p>
								</div>
								<Button
									class="h-7 shrink-0 gap-1 text-[11px]"
									disabled={webBusy}
									onclick={handleWebStage}
									size="sm"
								>
									<Inbox class="h-3.5 w-3.5" />
									<span>存入临时知识库</span>
								</Button>
							</div>
							<pre class="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-lg border border-border/60 bg-background/80 p-2 font-serif text-[11px] leading-relaxed text-foreground/85">{webPreview.preview}</pre>
						</div>
					{/if}

					<div class="flex gap-2 border-t border-border/60 pt-3">
						<Input
							bind:value={webQuery}
							class="text-xs"
							onkeydown={(e) => e.key === 'Enter' && handleWebSearchStage()}
							placeholder="联网搜索关键词（需配置搜索 Provider 或 MCP 搜索服务）…"
						/>
						<Button
							class="gap-1.5 shrink-0 text-xs"
							disabled={webBusy || !webStatus?.enabled || !webStatus?.search_available}
							onclick={handleWebSearchStage}
							size="sm"
							variant="secondary"
						>
							<Search class="h-3.5 w-3.5" />
							<span>搜索并暂存</span>
						</Button>
					</div>

					<p class="text-[11px] text-muted-foreground">
						说明：联网资料只会进入「临时知识库」，不会直接污染正式知识库；提纯成候选后由你确认晋升。
						网页抓取已做内网/回环地址拦截与大小限制。
					</p>
				</div>

			<!-- TAB 5: STAGING / INBOX -->
			{:else if activeTab === 'staging'}
				<div class="space-y-3">
					<div class="flex flex-wrap items-center gap-2">
						<Input
							bind:value={stagingCourseId}
							class="h-8 max-w-56 text-xs"
							placeholder="目标知识域 ID（默认 default）"
						/>
						<Button class="h-8 gap-1.5 text-xs" disabled={stagingBusy} onclick={loadStagingData} size="sm" variant="outline">
							<RefreshCw class="h-3.5 w-3.5 {stagingBusy ? 'animate-spin' : ''}" />
							<span>刷新临时库</span>
						</Button>
						<span class="text-[11px] text-muted-foreground">
							TTL {webStatus?.staging_ttl_days ?? 7} 天自动清理 · 晋升默认人工确认
						</span>
					</div>

					<!-- Staged documents -->
					<div class="space-y-2">
						<div class="flex items-center gap-1.5 text-[11px] font-semibold text-muted-foreground">
							<Inbox class="h-3.5 w-3.5 text-amber-500" />
							<span>待整理网页（{stagingDocuments.length}）</span>
						</div>
						{#if stagingDocuments.length === 0}
							<div class="rounded-xl border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
								暂无临时资料。可在「联网导入」抓取网页，或让聊天智能体在本地资料不足时自动补充。
							</div>
						{:else}
							{#each stagingDocuments as doc (doc.doc_id)}
								<div class="rounded-xl border border-border/80 bg-muted/20 p-2.5">
									<div class="flex flex-wrap items-center justify-between gap-2">
										<div class="min-w-0">
											<p class="truncate text-xs font-semibold text-foreground">{doc.title}</p>
											<div class="flex flex-wrap items-center gap-1.5 text-[10px] text-muted-foreground">
												<span class="rounded bg-indigo-500/10 px-1.5 py-0.5 font-mono text-indigo-600 dark:text-indigo-300">
													{doc.source_type === 'web' ? '🌐 网页' : '本地'}
												</span>
												<span class="rounded bg-muted px-1.5 py-0.5 font-mono">{doc.status}</span>
												{#if doc.quality_score > 0}
													<span class="rounded bg-emerald-500/10 px-1.5 py-0.5 font-mono text-emerald-600 dark:text-emerald-300">
														质量 {(doc.quality_score * 100).toFixed(0)}
													</span>
												{/if}
												<span class="truncate">{doc.url || doc.origin}</span>
											</div>
										</div>
										<div class="flex shrink-0 items-center gap-1">
											<Button
												class="h-7 gap-1 px-2 text-[11px]"
												disabled={stagingBusy || doc.status === 'promoted'}
												onclick={() => handleDistill(doc)}
												size="sm"
												variant="secondary"
											>
												<Sparkles class="h-3 w-3" />
												<span>提纯</span>
											</Button>
											{#if doc.status === 'promoted'}
												<Button
													class="h-7 gap-1 px-2 text-[11px]"
													disabled={stagingBusy}
													onclick={() => handleRevoke(doc)}
													size="sm"
													variant="outline"
												>
													<span>撤销晋升</span>
												</Button>
											{:else}
												<Button
													class="h-7 gap-1 px-2 text-[11px] text-muted-foreground"
													disabled={stagingBusy}
													onclick={() => handleDeleteStaging(doc)}
													size="sm"
													variant="ghost"
												>
													<Trash2 class="h-3 w-3" />
													<span>丢弃</span>
												</Button>
											{/if}
										</div>
									</div>
								</div>
							{/each}
						{/if}
					</div>

					<!-- Distilled candidates -->
					<div class="space-y-2 border-t border-border/60 pt-3">
						<div class="flex items-center justify-between gap-2">
							<div class="flex items-center gap-1.5 text-[11px] font-semibold text-muted-foreground">
								<Sparkles class="h-3.5 w-3.5 text-emerald-500" />
								<span>待入库知识候选（{stagingCandidates.length}）</span>
							</div>
							<Button
								class="h-7 gap-1 px-2 text-[11px]"
								disabled={promoting || selectedCandidateIds.length === 0}
								onclick={handlePromote}
								size="sm"
							>
								{#if promoting}
									<Loader2 class="h-3 w-3 animate-spin" />
								{:else}
									<CheckCircle2 class="h-3 w-3" />
								{/if}
								<span>晋升到正式库 ({selectedCandidateIds.length})</span>
							</Button>
						</div>
						{#if stagingCandidates.length === 0}
							<div class="rounded-xl border border-dashed border-border p-4 text-center text-xs text-muted-foreground">
								暂无待入库候选。对临时网页点击「提纯」后会生成知识节点候选。
							</div>
						{:else}
							{#each stagingCandidates as candidate (candidate.id)}
								<div class="flex items-start gap-2 rounded-xl border border-border/80 bg-card/60 p-2.5">
									<input
										aria-label="选择候选 {candidate.title}"
										checked={selectedCandidateIds.includes(candidate.id)}
										class="mt-1 h-3.5 w-3.5"
										onchange={() => toggleCandidate(candidate.id)}
										type="checkbox"
									/>
									<div class="min-w-0 flex-1">
										<div class="flex flex-wrap items-center gap-1.5">
											<span class="truncate text-xs font-semibold text-foreground">{candidate.title}</span>
											<span class="rounded bg-emerald-500/10 px-1.5 py-0.5 font-mono text-[10px] text-emerald-600 dark:text-emerald-300">
												质量 {(candidate.quality_score * 100).toFixed(0)}
											</span>
											{#if candidate.source_title}
												<span class="truncate text-[10px] text-muted-foreground">来自：{candidate.source_title}</span>
											{/if}
										</div>
										<p class="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-muted-foreground">
											{candidate.summary}
										</p>
									</div>
									<Button
										class="h-7 shrink-0 px-2 text-[11px] text-muted-foreground"
										onclick={() => handleRejectCandidate(candidate)}
										size="sm"
										variant="ghost"
									>
										驳回
									</Button>
								</div>
							{/each}
						{/if}
					</div>
				</div>
			{/if}
		</div>
	</Dialog.Content>
</Dialog.Root>
