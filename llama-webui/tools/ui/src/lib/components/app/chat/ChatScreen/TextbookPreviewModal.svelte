<script lang="ts">
	import { ChevronLeft, ChevronRight, Loader2, X } from '@lucide/svelte';
	import { provenanceNavigation } from '$lib/services/provenance-navigation.service';
	import { getActiveRagNamespace, learningHeaders } from '$lib/services/learning-context';
	import { getPDFPageCount, renderPDFPageToImage } from '$lib/utils/pdf-processing';

	let loading = $state(false);
	let error = $state<string | null>(null);
	let pageImage = $state<string | null>(null);
	let pageText = $state<string | null>(null);
	let totalPages = $state<number | null>(null);
	let currentPage = $state(1);
	let activeDocId = $state<string | null>(null);
	let requestToken = 0;

	function isPdfName(name?: string): boolean {
		return !!name && name.toLowerCase().endsWith('.pdf');
	}

	function close(): void {
		requestToken += 1;
		provenanceNavigation.closeTextbookPreview();
	}

	async function loadPage(page: number): Promise<void> {
		const target = provenanceNavigation.textbookViewerTarget;
		if (!target || !target.docId) return;

		const token = ++requestToken;
		loading = true;
		error = null;
		activeDocId = target.docId;

		const namespace = target.namespace || getActiveRagNamespace();
		const resolvedPage = Math.max(1, Math.floor(page) || 1);
		currentPage = resolvedPage;

		try {
			const base = `/api/rag/documents/${encodeURIComponent(target.docId)}`;
			const namespaceQuery = `namespace=${encodeURIComponent(namespace)}`;

			if (isPdfName(target.docTitle)) {
				const fileResponse = await fetch(`${base}/file?${namespaceQuery}`, {
					headers: learningHeaders()
				});
				if (!fileResponse.ok) {
					throw new Error(`无法读取原始文档（HTTP ${fileResponse.status}）`);
				}
				const blob = await fileResponse.blob();
				const file = new File([blob], target.docTitle || 'document.pdf', {
					type: blob.type || 'application/pdf'
				});
				const count = await getPDFPageCount(file);
				const image = await renderPDFPageToImage(file, resolvedPage);
				if (token !== requestToken) return;
				totalPages = count;
				pageImage = image;
				pageText = null;
			} else {
				const pageResponse = await fetch(
					`${base}/page?${namespaceQuery}&page=${resolvedPage}`,
					{ headers: learningHeaders() }
				);
				if (!pageResponse.ok) {
					throw new Error(`无法读取该页原文（HTTP ${pageResponse.status}）`);
				}
				const data = (await pageResponse.json()) as {
					page?: number;
					total_pages?: number;
					text?: string;
				};
				if (token !== requestToken) return;
				totalPages = data.total_pages ?? null;
				pageText = data.text ?? '';
				pageImage = null;
			}
		} catch (err) {
			if (token !== requestToken) return;
			pageImage = null;
			pageText = null;
			error = err instanceof Error ? err.message : String(err);
		} finally {
			if (token === requestToken) loading = false;
		}
	}

	function goToPage(delta: number): void {
		const next = currentPage + delta;
		if (totalPages !== null && (next < 1 || next > totalPages)) return;
		if (next < 1) return;
		void loadPage(next);
	}

	$effect(() => {
		const open = provenanceNavigation.textbookViewerOpen;
		const target = provenanceNavigation.textbookViewerTarget;

		if (open && target) {
			if (activeDocId !== target.docId) {
				totalPages = null;
				pageImage = null;
				pageText = null;
				error = target.docId
					? null
					: '该知识节点缺少结构化文档编号（旧数据），无法定位原文，仅展示摘录内容。';
				if (target.docId) void loadPage(target.pageStart || 1);
			}
		} else {
			activeDocId = null;
		}
	});

	function highlightParts(text: string, quote?: string): Array<{ text: string; mark: boolean }> {
		const needle = (quote || '').trim();
		if (!needle) return [{ text, mark: false }];
		const idx = text.indexOf(needle);
		if (idx < 0) return [{ text, mark: false }];
		return [
			{ text: text.slice(0, idx), mark: false },
			{ text: needle, mark: true },
			{ text: text.slice(idx + needle.length), mark: false }
		];
	}
</script>

{#if provenanceNavigation.textbookViewerOpen && provenanceNavigation.textbookViewerTarget}
	{@const target = provenanceNavigation.textbookViewerTarget}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs"
		role="dialog"
		aria-modal="true"
		aria-label="文档溯源预览"
	>
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div
			class="relative flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
			onmousedown={(e) => e.stopPropagation()}
		>
			<div class="flex items-center justify-between border-b border-border/80 bg-muted/40 px-4 py-3">
				<div class="flex min-w-0 items-center gap-2">
					<span class="text-base">📖</span>
					<div class="min-w-0">
						<h3 class="truncate text-sm font-semibold text-foreground">
							{target.docTitle || '知识源文档'}
						</h3>
						<p class="font-mono text-[11px] text-muted-foreground">
							物理页码: P.{currentPage}{totalPages && totalPages > 1
								? ` / 共 ${totalPages} 页`
								: ''}
						</p>
					</div>
				</div>
				<div class="flex items-center gap-1">
					<button
						aria-label="上一页"
						class="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
						disabled={loading || currentPage <= 1}
						onclick={() => goToPage(-1)}
						type="button"
					>
						<ChevronLeft class="h-4 w-4" />
					</button>
					<button
						aria-label="下一页"
						class="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
						disabled={loading || (totalPages !== null && currentPage >= totalPages)}
						onclick={() => goToPage(1)}
						type="button"
					>
						<ChevronRight class="h-4 w-4" />
					</button>
					<button
						aria-label="关闭预览"
						class="rounded-md p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
						onclick={close}
						title="关闭预览"
						type="button"
					>
						<X class="h-4 w-4" />
					</button>
				</div>
			</div>

			<div class="flex-1 space-y-3 overflow-y-auto p-4">
				{#if target.quoteSnippet}
					<div class="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3">
						<div class="mb-1 text-xs font-semibold text-emerald-700 dark:text-emerald-300">
							🎯 划词摘录
						</div>
						<blockquote
							class="select-text break-words font-serif text-xs leading-relaxed text-foreground/90"
						>
							“{target.quoteSnippet}”
						</blockquote>
					</div>
				{/if}

				{#if loading}
					<div class="flex items-center justify-center gap-2 p-8 text-xs text-muted-foreground">
						<Loader2 class="h-4 w-4 animate-spin" />
						正在读取文档原文…
					</div>
				{:else if error}
					<div class="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-amber-700 dark:text-amber-300">
						{error}
					</div>
				{:else if pageImage}
					<div class="flex flex-col items-center gap-2">
						<img
							alt={`{target.docTitle || '文档'} 第 ${currentPage} 页`}
							class="w-full rounded-lg border border-border/60 shadow"
							src={pageImage}
						/>
						<p class="font-mono text-[10px] text-muted-foreground">
							第 {currentPage} 页 · pdf.js 渲染
						</p>
					</div>
				{:else if pageText}
					<div
						class="rounded-xl border border-border/80 bg-muted/20 p-4 font-serif text-xs leading-6 text-foreground/90 whitespace-pre-wrap"
					>
						{#each highlightParts(pageText, target.quoteSnippet) as part}
							{#if part.mark}
								<mark class="rounded bg-primary/20 px-0.5 text-foreground">{part.text}</mark>
							{:else}
								{part.text}
							{/if}
						{/each}
					</div>
				{:else if target.docId}
					<div class="p-4 text-xs text-muted-foreground">该页没有可显示的文本内容。</div>
				{/if}
			</div>

			<div class="flex items-center justify-between border-t border-border/80 bg-muted/20 px-4 py-2.5">
				<span class="text-[11px] text-muted-foreground">
					来源：{target.docTitle || '知识源文档'} · P.{currentPage}
				</span>
				<button
					class="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-accent"
					onclick={close}
					type="button"
				>
					关闭
				</button>
			</div>
		</div>
	</div>
{/if}
