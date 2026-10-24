<script lang="ts">
	import {
		ChatMessageActionIcons,
		ChatMessageAgenticContent,
		ChatMessageAssistantModel,
		ChatMessageAssistantProcessingInfo,
		ChatMessageAssistantRawOutput,
		ChatMessageAssistantStatistics,
		ChatMessageEditForm
	} from '$lib/components/app';
	import { getChatMessageEditContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { provenanceNavigation } from '$lib/services/provenance-navigation.service';
	import {
		chatStore,
		modelsStore,
		retrievalStatusStore,
		serverStore,
		settingsStore
	} from '$lib/stores';
	import { hasAgenticContent, modelLoadProgressText } from '$lib/utils';
	import { Brain, Sparkles } from '@lucide/svelte';

	interface Props {
		class?: string;
		isLastAssistantMessage?: boolean;
		message: DatabaseMessage;
		toolMessages?: DatabaseMessage[];
		onContinue?: () => void;
		onRegenerate: (modelOverride?: string) => void;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		isLastAssistantMessage = false,
		message,
		onContinue,
		onRegenerate,
		textareaElement = $bindable(),
		toolMessages = []
	}: Props = $props();

	// Get edit context
	const editCtx = getChatMessageEditContext();

	const isAgentic = $derived(hasAgenticContent(message, toolMessages));
	const processingState = useProcessingState();

	let currentConfig = $derived(settingsStore.config);
	let isRouter = $derived(serverStore.isRouterMode);

	let showRawOutput = $state(false);

	let displayedModel = $derived(message.model ?? null);

	let isCurrentlyLoading = $derived(chatStore.isLoading);
	let isStreaming = $derived(chatStore.isStreaming());
	let hasNoContent = $derived(!message?.content?.trim());
	let isActivelyProcessing = $derived(isCurrentlyLoading || isStreaming);

	// during a router auto-load the message has no model yet: target the model frozen in the
	// persisted stream state (survives a reload), then fall back to the dropdown selection
	let loadTargetModel = $derived(
		message.model ?? chatStore.getResumeModel(message.convId) ?? modelsStore.selectedModelName
	);
	let modelLoadProgress = $derived(
		isRouter && loadTargetModel ? modelsStore.status.getLoadProgress(loadTargetModel) : null
	);
	let modelLoadingText = $derived(modelLoadProgressText(modelLoadProgress));

	let showProcessingInfoTop = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			hasNoContent &&
			!isAgentic &&
			isLastAssistantMessage
	);

	let showProcessingInfoBottom = $derived(
		message?.role === MessageRole.ASSISTANT &&
			isActivelyProcessing &&
			(!hasNoContent || isAgentic) &&
			isLastAssistantMessage
	);

	let associatedKnowledgeNodes = $derived(
		provenanceNavigation.getKnowledgeNodesForMessage(message.id)
	);

	const RETRIEVAL_MODE_LABELS: Record<string, string> = {
		explain: '讲解模式',
		quiz: '练习模式',
		grade: '判分模式',
		review: '复习模式',
		plan: '规划模式',
		homework: '作业辅导',
		chat: '对话模式'
	};

	let retrievalStatus = $derived(
		isLastAssistantMessage && retrievalStatusStore.current?.conversationId === message.convId
			? retrievalStatusStore.current
			: null
	);

	let retrievalSummary = $derived.by(() => {
		if (!retrievalStatus) return '';

		const web =
			retrievalStatus.webSources > 0 ? ` · 联网 ${retrievalStatus.webSources} 条` : '';

		return `本轮检索 · RAG ${retrievalStatus.ragHits} 段${web}`;
	});

	let assistantEl: HTMLDivElement | undefined = $state();
	let lastUserMessageHeight = $state(0);
	let assistantMarginTop = $state(0);

	// The measured CSS vars feed the :last-child min-height rule only, so only
	// the last assistant message needs them. Reading isLastAssistantMessage
	// here also re-runs the effect when this message stops being the last.
	$effect(() => {
		if (!assistantEl || !isLastAssistantMessage) return;

		assistantMarginTop = Math.round(parseFloat(getComputedStyle(assistantEl).marginTop));

		const chatMessageEl = assistantEl.closest('.chat-message');
		const previousChatMessage = chatMessageEl?.previousElementSibling;
		const userMessageEl = previousChatMessage?.querySelector(
			'.chat-message-user'
		) as HTMLElement | null;

		if (!userMessageEl) {
			lastUserMessageHeight = 0;

			return;
		}

		const updateHeight = () => {
			const rect = userMessageEl.getBoundingClientRect();
			const marginTop = Math.round(parseFloat(getComputedStyle(userMessageEl).marginTop));

			lastUserMessageHeight = Math.round(rect.height + marginTop);
		};

		updateHeight();

		const resizeObserver = new ResizeObserver(updateHeight);

		resizeObserver.observe(userMessageEl);

		return () => {
			resizeObserver.disconnect();
		};
	});

	$effect(() => {
		if (showProcessingInfoTop || showProcessingInfoBottom) {
			processingState.startMonitoring();
		}
	});
</script>

<div
	bind:this={assistantEl}
	style:--assistant-margin-top={assistantMarginTop > 0 ? `${assistantMarginTop}px` : undefined}
	style:--last-user-message-height={lastUserMessageHeight > 0
		? `${lastUserMessageHeight}px`
		: undefined}
	aria-label="Assistant message with actions"
	class="chat-message-assistant text-md group w-full leading-7.5 {className}"
	role="group"
>
	{#if showProcessingInfoTop}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} position="top" {processingState} />
	{/if}

	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		{#if showRawOutput}
			<ChatMessageAssistantRawOutput {message} {toolMessages} />
		{:else}
			<ChatMessageAgenticContent
				{isLastAssistantMessage}
				isStreaming={chatStore.isStreaming()}
				{message}
				{toolMessages}
			/>
		{/if}
	{/if}

	{#if showProcessingInfoBottom}
		<ChatMessageAssistantProcessingInfo {modelLoadingText} position="bottom" {processingState} />
	{/if}

	{#if associatedKnowledgeNodes.length > 0}
		<div class="knowledge-capsules mt-2.5 mb-2 flex flex-wrap items-center gap-1.5 pt-1">
			<span class="text-[11px] text-muted-foreground flex items-center gap-1">
				<Brain class="h-3 w-3 text-primary" />
				已沉淀知识点:
			</span>
			{#each associatedKnowledgeNodes as knode (knode.id)}
				<button
					type="button"
					class="group inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 hover:bg-primary/20 px-2.5 py-0.5 text-xs font-medium text-primary shadow-xs transition-all hover:scale-105 active:scale-95 cursor-pointer"
					onclick={() => provenanceNavigation.focusCanvasNode(knode.id)}
					title="在画布中居中聚焦卡片「{knode.title}」"
				>
					<Sparkles class="h-3 w-3 text-primary group-hover:rotate-12 transition-transform" />
					<span class="max-w-48 truncate">{knode.title}</span>
				</button>
			{/each}
		</div>
	{/if}

	{#if retrievalStatus}
		<div
			class="retrieval-status mt-2 mb-1 flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground"
		>
			<span
				class="inline-flex items-center gap-1 rounded-full border border-border/70 bg-muted/40 px-2 py-0.5"
			>
				{retrievalSummary}
			</span>
			<span
				class="inline-flex items-center gap-1 rounded-full border border-border/70 bg-muted/40 px-2 py-0.5"
			>
				{retrievalStatus.retrievalMode === 'none' ? '未触发检索' : '本地知识库'}
			</span>
			<span
				class="inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-primary"
			>
				{RETRIEVAL_MODE_LABELS[retrievalStatus.mode] ?? retrievalStatus.mode}
			</span>
		</div>
	{/if}

	{#if displayedModel}
		<div class="info my-6 grid gap-4 tabular-nums">
			<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
				<ChatMessageAssistantModel
					{displayedModel}
					isLoading={chatStore.isLoading}
					{isRouter}
					{onRegenerate}
				/>

				<ChatMessageAssistantStatistics
					isLoading={chatStore.isLoading}
					{message}
					{processingState}
					showMessageStats={currentConfig.showMessageStats}
				/>
			</div>
		</div>
	{/if}

	{#if message.timestamp && !editCtx.isEditing}
		<ChatMessageActionIcons
			actionsPosition="left"
			justify="start"
			onContinue={currentConfig.enableContinueGeneration ? onContinue : undefined}
			onRawOutputToggle={(enabled) => (showRawOutput = enabled)}
			{onRegenerate}
			rawOutputEnabled={showRawOutput}
			role={MessageRole.ASSISTANT}
			showRawOutputSwitch={currentConfig.showRawOutputSwitch}
		/>
	{/if}
</div>

<style>
	:global(.chat-message):last-child .chat-message-assistant {
		--assistant-min-height-offset: calc(
			var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
				var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
				var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
		);
		min-height: calc(100dvh - var(--assistant-min-height-offset));

		@media (width > 768px) {
			--assistant-min-height-offset: calc(
				var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
					var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
					var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
			);
		}
	}
</style>
