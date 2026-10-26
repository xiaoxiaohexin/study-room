<script lang="ts">
	import LazyChatMessage from './LazyChatMessage.svelte';
	import { ChatMessageUserPending } from '$lib/components/app';
	import { MessageRole } from '$lib/enums';
	import {
		extractTitleCandidate,
		provenanceNavigation
	} from '$lib/services/provenance-navigation.service';
	import { attributeQuote } from '$lib/services/rag-attribution.service';
	import { getActiveCourseId, getActiveRagNamespace } from '$lib/services/learning-context';
	import { agenticStore, chatStore, conversationsStore, settingsStore } from '$lib/stores';
	import type { ChatMessageActions } from '$lib/types';
	import type { CustomCanvasNode } from '$lib/types/canvas';
	import {
		buildSiblingInfoMap,
		buildTurnIndexMap,
		copyToClipboard,
		formatMessageForClipboard,
		hasAgenticContent
	} from '$lib/utils';
	import { ChevronDown, Loader2, PlusCircle, Sparkles, X } from '@lucide/svelte';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	interface Props {
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
		onMessagesReady?: (messageCount: number) => void;
	}

	let { messages = [], onMessagesReady, onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);

	const currentConfig = settingsStore.config;

	const chatActions: ChatMessageActions = {
		continueAssistantMessage: async (message: DatabaseMessage) => {
			onUserAction?.();
			await chatStore.continueAssistantMessage(message.id);
			refreshAllMessages();
		},

		copy: async (message: DatabaseMessage, contentOverride?: string) => {
			const asPlainText = Boolean(currentConfig.copyTextAttachmentsAsPlainText);
			const clipboardContent = formatMessageForClipboard(
				contentOverride ?? message.content,
				message.extra,
				asPlainText
			);

			await copyToClipboard(clipboardContent, 'Message copied to clipboard');
		},

		delete: async (message: DatabaseMessage) => {
			await chatStore.deleteMessage(message.id);
			refreshAllMessages();
		},

		editUserMessagePreserveResponses: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			// in-place edit: the store already updated activeMessages and no
			// branch is created, so sibling info stays valid without a refetch
			await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);
		},

		editWithBranching: async (
			message: DatabaseMessage,
			newContent: string,
			newExtras?: DatabaseMessageExtra[]
		) => {
			onUserAction?.();
			await chatStore.editMessageWithBranching(message.id, newContent, newExtras);
			refreshAllMessages();
		},

		editWithReplacement: async (
			message: DatabaseMessage,
			newContent: string,
			shouldBranch: boolean
		) => {
			onUserAction?.();
			await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);

			// only a branch changes sibling info; an in-place edit already
			// landed in activeMessages
			if (shouldBranch) refreshAllMessages();
		},

		forkConversation: async (
			message: DatabaseMessage,
			options: { name: string; includeAttachments: boolean }
		) => {
			await conversationsStore.forkConversation(message.id, options);
		},

		navigateToSibling: async (siblingId: string) => {
			await conversationsStore.navigateToSibling(siblingId);
		},

		regenerateWithBranching: async (message: DatabaseMessage, modelOverride?: string) => {
			onUserAction?.();
			await chatStore.regenerateMessageWithBranching(message.id, modelOverride);
			refreshAllMessages();
		}
	};

	function refreshAllMessages() {
		const conversation = conversationsStore.activeConversation;

		if (conversation) {
			// reuse the array loadConversation just read, when present; branch
			// actions fall through to a fresh fetch
			const preloaded = conversationsStore.consumeLastLoadedMessages(conversation.id);

			if (preloaded) {
				allConversationMessages = preloaded;
			} else {
				conversationsStore.getConversationMessages(conversation.id).then((messages) => {
					allConversationMessages = messages;
				});
			}
		} else {
			allConversationMessages = [];
		}
	}

	// Refresh messages whenever the active conversation changes
	$effect(() => {
		if (conversationsStore.activeConversation) {
			refreshAllMessages();
		}
	});

	$effect(() => {
		void allConversationMessages;

		onMessagesReady?.(displayMessages.length);
	});

	let siblingInfoByMessageId = $derived(buildSiblingInfoMap(allConversationMessages));

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		const filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== MessageRole.SYSTEM);
		// Build display entries, grouping agentic sessions into single entries.
		// An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
		const result: Array<{
			message: DatabaseMessage;
			toolMessages: DatabaseMessage[];
			isLastAssistantMessage: boolean;
			isLastUserMessage: boolean;
			nextAssistantMessage: DatabaseMessage | null;
			siblingInfo: ChatMessageSiblingInfo;
		}> = [];

		for (let i = 0; i < filteredMessages.length; i++) {
			const msg = filteredMessages[i];

			// Skip tool messages - they're grouped with preceding assistant
			if (msg.role === MessageRole.TOOL) continue;

			const toolMessages: DatabaseMessage[] = [];

			if (msg.role === MessageRole.ASSISTANT && hasAgenticContent(msg)) {
				let j = i + 1;

				while (j < filteredMessages.length) {
					const next = filteredMessages[j];

					if (next.role === MessageRole.TOOL) {
						toolMessages.push(next);

						j++;
					} else if (next.role === MessageRole.ASSISTANT) {
						toolMessages.push(next);

						j++;
					} else {
						break;
					}
				}

				i = j - 1;
			} else if (msg.role === MessageRole.ASSISTANT) {
				let j = i + 1;

				while (j < filteredMessages.length && filteredMessages[j].role === MessageRole.TOOL) {
					toolMessages.push(filteredMessages[j]);
					j++;
				}
			}

			const siblingInfo = siblingInfoByMessageId.get(msg.id) ?? {
				currentIndex: 0,
				message: msg,
				siblingIds: [msg.id],
				totalSiblings: 1
			};

			result.push({
				isLastAssistantMessage: false,
				isLastUserMessage: false,
				message: msg,
				nextAssistantMessage: null,
				siblingInfo,
				toolMessages
			});
		}

		let lastAssistantIdx = -1;

		for (let i = result.length - 1; i >= 0; i--) {
			if (result[i].message.role === MessageRole.ASSISTANT) {
				result[i].isLastAssistantMessage = true;
				lastAssistantIdx = i;

				break;
			}
		}

		if (lastAssistantIdx > 0 && result[lastAssistantIdx - 1].message.role === MessageRole.USER) {
			result[lastAssistantIdx - 1].isLastUserMessage = true;
		}

		for (let i = 0; i < result.length; i++) {
			if (result[i].message.role !== MessageRole.USER) continue;

			for (let j = i + 1; j < result.length; j++) {
				if (result[j].message.role === MessageRole.ASSISTANT) {
					result[i].nextAssistantMessage = result[j].message;

					break;
				}
			}
		}

		return result;
	});

	// Real conversation turn numbers (1-based). The sibling index exposed by
	// buildSiblingInfoMap is a branch index, not a turn number.
	let turnIndexByMessageId = $derived.by(() =>
		buildTurnIndexMap(
			displayMessages.map((entry) => ({ id: entry.message.id, role: entry.message.role }))
		)
	);

	let toolbarVisible = $state(false);
	let toolbarCoords = $state({ x: 0, y: 0, placeBelow: false });
	let selectedText = $state('');
	let originMessageId = $state('');
	let originTurnIndex = $state(1);
	let originMessageRole = $state('assistant');
	let isExtracting = $state(false);
	let showAppendDropdown = $state(false);

	function handleMouseUp() {
		setTimeout(checkTextSelection, 20);
	}

	function checkTextSelection() {
		if (typeof window === 'undefined') return;
		const selection = window.getSelection();
		if (!selection || selection.isCollapsed) {
			if (!showAppendDropdown) toolbarVisible = false;
			return;
		}

		const rawText = selection.toString();
		const trimmedText = rawText.trim();
		if (trimmedText.length < 2) {
			if (!showAppendDropdown) toolbarVisible = false;
			return;
		}

		const anchorNode = selection.anchorNode;
		const anchorEl = anchorNode instanceof Element ? anchorNode : anchorNode?.parentElement;
		const startMsgEl = anchorEl?.closest('[data-message-id]');

		if (!startMsgEl) {
			if (!showAppendDropdown) toolbarVisible = false;
			return;
		}

		const role = startMsgEl.getAttribute('data-message-role') || 'assistant';
		if (role !== 'assistant') {
			if (!showAppendDropdown) toolbarVisible = false;
			return;
		}

		const msgId = startMsgEl.getAttribute('data-message-id') || '';
		const turnIdxStr = startMsgEl.getAttribute('data-turn-index');
		const turnIdx = turnIdxStr ? parseInt(turnIdxStr, 10) : 1;

		selectedText = trimmedText;
		originMessageId = msgId;
		originTurnIndex = turnIdx;
		originMessageRole = role;

		try {
			const range = selection.getRangeAt(0);
			const rect = range.getBoundingClientRect();
			const toolbarWidth = 280;
			const toolbarHeight = 44;
			const margin = 10;

			let x = rect.left + rect.width / 2;
			let y = rect.top - margin;
			let placeBelow = false;

			if (y - toolbarHeight < 12) {
				y = rect.bottom + margin;
				placeBelow = true;
			}

			const minX = toolbarWidth / 2 + 12;
			const maxX = window.innerWidth - toolbarWidth / 2 - 12;
			x = Math.max(minX, Math.min(maxX, x));

			toolbarCoords = { x, y, placeBelow };
			toolbarVisible = true;
		} catch {
			toolbarVisible = false;
		}
	}

	function handleSelectionChange() {
		if (typeof window === 'undefined') return;
		const selection = window.getSelection();
		if (!selection || selection.isCollapsed) {
			if (!showAppendDropdown) {
				toolbarVisible = false;
			}
		}
	}

	async function handleExtractKnowledgeNode() {
		if (!selectedText || isExtracting) return;
		isExtracting = true;

		try {
			const title = extractTitleCandidate(selectedText);
			const courseId = getActiveCourseId();
			const namespace = getActiveRagNamespace();

			// Only attach a textbook anchor when the backend can really attribute the
			// quote to a chunk/page. Never fabricate a fallback "reference textbook P.1".
			const attribution = await attributeQuote(selectedText, namespace);
			const matched = attribution?.matched === true && !!attribution.chunk;

			const eventPayload = {
				title,
				quoteSnippet: selectedText,
				messageId: originMessageId,
				turnIndex: originTurnIndex,
				courseId,
				docId: matched ? attribution!.chunk!.doc_id : undefined,
				chunkId: matched ? attribution!.chunk!.chunk_id : undefined,
				docTitle: matched ? attribution!.chunk!.filename : undefined,
				pageStart: matched ? attribution!.chunk!.page_start : undefined,
				pageEnd: matched ? attribution!.chunk!.page_end : undefined,
				difficulty: 0.5,
				mastery: 0.3
			};

			document.dispatchEvent(
				new CustomEvent('canvas:create-knowledge-node', {
					detail: eventPayload
				})
			);

			toast.success(
				matched
					? `已提取知识节点「${title}」，并匹配到文档出处 P.${eventPayload.pageStart}`
					: `已提取知识节点「${title}」（未匹配到文档出处，仅保留对话锚点）`
			);
			toolbarVisible = false;
			showAppendDropdown = false;
			window.getSelection()?.removeAllRanges();
		} catch (err) {
			console.error('Failed to extract knowledge node:', err);
			toast.error('提取知识节点失败');
		} finally {
			isExtracting = false;
		}
	}

	async function handleAppendToTargetNode(targetNode: CustomCanvasNode) {
		if (!selectedText || !targetNode) return;
		const courseId = getActiveCourseId();
		const namespace = getActiveRagNamespace();

		try {
			const attribution = await attributeQuote(selectedText, namespace);
			const matched = attribution?.matched === true && !!attribution.chunk;

			document.dispatchEvent(
				new CustomEvent('canvas:append-to-knowledge-node', {
					detail: {
						targetNodeId: targetNode.id,
						courseId,
						quoteSnippet: selectedText,
						messageId: originMessageId,
						turnIndex: originTurnIndex,
						docId: matched ? attribution!.chunk!.doc_id : undefined,
						chunkId: matched ? attribution!.chunk!.chunk_id : undefined,
						docTitle: matched ? attribution!.chunk!.filename : undefined,
						pageStart: matched ? attribution!.chunk!.page_start : undefined,
						pageEnd: matched ? attribution!.chunk!.page_end : undefined
					}
				})
			);

			toast.success(
				matched
					? `已追加引用到节点「${targetNode.title}」，并匹配到文档出处 P.${attribution!.chunk!.page_start}`
					: `已追加引用到节点「${targetNode.title}」（未匹配到文档出处）`
			);
			toolbarVisible = false;
			showAppendDropdown = false;
			window.getSelection()?.removeAllRanges();
		} catch (err) {
			console.error('Failed to append to knowledge node:', err);
			toast.error('追加引用失败');
		}
	}

	onMount(() => {
		document.addEventListener('selectionchange', handleSelectionChange);
		const handleDocumentClick = (e: MouseEvent) => {
			const target = e.target as HTMLElement | null;
			if (target && !target.closest('.selection-floating-toolbar')) {
				if (showAppendDropdown) showAppendDropdown = false;
				const sel = window.getSelection();
				if (!sel || sel.isCollapsed) toolbarVisible = false;
			}
		};
		const handleDocumentKeyDown = (e: KeyboardEvent) => {
			if (e.key === 'Escape') {
				toolbarVisible = false;
				showAppendDropdown = false;
			}
		};
		document.addEventListener('mousedown', handleDocumentClick);
		document.addEventListener('keydown', handleDocumentKeyDown);

		return () => {
			document.removeEventListener('selectionchange', handleSelectionChange);
			document.removeEventListener('mousedown', handleDocumentClick);
			document.removeEventListener('keydown', handleDocumentKeyDown);
		};
	});
</script>

<!-- Re-created per conversation, so the CSS fade-in below plays on every
     navigation into a chat route. -->
{#key conversationsStore.activeConversation?.id ?? 'new'}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div class="chat-messages" onmouseup={handleMouseUp} ontouchend={handleMouseUp}>
		{#each displayMessages as { isLastAssistantMessage, isLastUserMessage, message, nextAssistantMessage, siblingInfo, toolMessages } (message.id)}
			<LazyChatMessage
				{chatActions}
				class="mx-auto mt-12 w-full max-w-3xl"
				{isLastAssistantMessage}
				{isLastUserMessage}
				{message}
				{nextAssistantMessage}
				{siblingInfo}
				{toolMessages}
				turnIndex={turnIndexByMessageId.get(message.id)}
			/>
		{/each}

		{#if conversationsStore.activeConversation && agenticStore.getPendingSteeringMessageContent(conversationsStore.activeConversation!.id)}
			{@const convId = conversationsStore.activeConversation!.id}
			{@const pendingContent = agenticStore.getPendingSteeringMessageContent(convId)}

			{#if pendingContent}
				<ChatMessageUserPending
					class="mx-auto mt-12 w-full max-w-[48rem]"
					content={pendingContent}
					extras={agenticStore.getPendingSteeringMessageExtras(convId)}
					onDelete={() => agenticStore.clearSteeringMessage(convId)}
					onEdit={(newContent, extras) =>
						agenticStore.injectSteeringMessage(convId, newContent, extras)}
					onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				/>
			{/if}
		{:else if conversationsStore.activeConversation && chatStore.getPendingMessageContent(conversationsStore.activeConversation!.id)}
			{@const convId = conversationsStore.activeConversation!.id}
			{@const pendingContent = chatStore.getPendingMessageContent(convId)}

			{#if pendingContent}
				<ChatMessageUserPending
					class="mx-auto mt-12 w-full max-w-[48rem]"
					content={pendingContent}
					extras={chatStore.getPendingMessageExtras(convId)}
					onDelete={() => chatStore.clearPendingMessage(convId)}
					onEdit={(newContent, extras) =>
						chatStore.injectPendingMessage(convId, newContent, extras)}
					onSendImmediately={() => chatStore.abortCurrentFlow(convId)}
				/>
			{/if}
		{/if}
	</div>
{/key}

{#if toolbarVisible}
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="selection-floating-toolbar fixed z-50 flex flex-col items-center select-none"
		style="left: {toolbarCoords.x}px; top: {toolbarCoords.y}px; transform: translate(-50%, {toolbarCoords.placeBelow ? '0%' : '-100%'});"
		onmousedown={(e) => e.stopPropagation()}
	>
		<div
			class="flex items-center gap-1 p-1 rounded-xl bg-popover text-popover-foreground shadow-2xl border border-border/80 backdrop-blur-md"
		>
			<button
				type="button"
				class="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold text-amber-700 dark:text-amber-300 bg-amber-500/15 hover:bg-amber-500/25 transition-colors cursor-pointer disabled:opacity-50"
				onclick={handleExtractKnowledgeNode}
				disabled={isExtracting}
				title="将选中文本提炼为新的知识节点"
			>
				{#if isExtracting}
					<Loader2 class="h-3.5 w-3.5 animate-spin text-amber-500" />
					<span>提取中…</span>
				{:else}
					<Sparkles class="h-3.5 w-3.5 text-amber-500" />
					<span>提取为知识节点</span>
				{/if}
			</button>

			<div class="h-4 w-px bg-border/60 mx-0.5"></div>

			<button
				type="button"
				class="inline-flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors cursor-pointer"
				onclick={() => (showAppendDropdown = !showAppendDropdown)}
				title="将当前摘录追加到画布已有的知识卡片"
			>
				<PlusCircle class="h-3.5 w-3.5" />
				<span>追加到已有节点</span>
				<ChevronDown class="h-3 w-3 opacity-70 transition-transform {showAppendDropdown ? 'rotate-180' : ''}" />
			</button>

			<button
				type="button"
				class="p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors cursor-pointer"
				onclick={() => { toolbarVisible = false; showAppendDropdown = false; }}
				title="关闭"
			>
				<X class="h-3 w-3" />
			</button>
		</div>

		{#if showAppendDropdown}
			<div
				class="mt-1.5 w-64 max-h-56 overflow-y-auto rounded-xl border border-border bg-popover p-1.5 shadow-2xl backdrop-blur-md"
			>
				<div class="px-2 py-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider border-b border-border/50 mb-1">
					选择目标知识卡片
				</div>
				{#if provenanceNavigation.knowledgeNodes.length > 0}
					<div class="space-y-0.5">
						{#each provenanceNavigation.knowledgeNodes as knode (knode.id)}
							<button
								type="button"
								class="w-full text-left px-2 py-1.5 rounded-lg hover:bg-accent transition-colors flex items-center justify-between group cursor-pointer"
								onclick={() => handleAppendToTargetNode(knode)}
							>
								<div class="min-w-0 flex-1 pr-2">
									<div class="text-xs font-medium text-foreground truncate group-hover:text-primary">
										{knode.title}
									</div>
									<div class="text-[10px] text-muted-foreground truncate">
										{knode.rawQuote || knode.content || '无摘录'}
									</div>
								</div>
								<span class="shrink-0 text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/10 text-primary">
									v{knode.currentVersion ?? 1}
								</span>
							</button>
						{/each}
					</div>
				{:else}
					<div class="p-3 text-center text-xs text-muted-foreground">
						画布暂无知识节点，请先「提取为知识节点」
					</div>
				{/if}
			</div>
		{/if}
	</div>
{/if}

<style>
	/* Compositor-friendly opacity fade; the keyed block re-creates the list per
	 * conversation, so the animation plays on every navigation into a chat. */
	.chat-messages {
		animation: chat-messages-fade-in 150ms ease-out;
	}

	@keyframes chat-messages-fade-in {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}

	:global(.pulse-highlight-target) {
		animation: provenance-message-pulse 2.5s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
	}

	@keyframes provenance-message-pulse {
		0% {
			outline: 3px solid var(--color-primary, #6366f1);
			outline-offset: 4px;
			background-color: color-mix(in oklab, var(--color-primary, #6366f1) 22%, transparent);
			transform: scale(1.012);
		}
		30% {
			outline: 3px solid var(--color-primary, #6366f1);
			outline-offset: 4px;
			background-color: color-mix(in oklab, var(--color-primary, #6366f1) 15%, transparent);
		}
		100% {
			outline: 3px solid transparent;
			outline-offset: 0px;
			background-color: transparent;
			transform: scale(1);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.chat-messages {
			animation: none;
		}
	}
</style>
