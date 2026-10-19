<script lang="ts">
	import { ArrowRight, Copy, Edit, GitBranch, RefreshCw, Trash2 } from '@lucide/svelte';
	import {
		ActionIcon,
		ChatMessageActionIconsBranchingControls,
		DialogConfirmation
	} from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { getChatMessageActionsContext, getChatMessageEditContext } from '$lib/contexts';
	import { MessageRole } from '$lib/enums';
	import { conversationsStore } from '$lib/stores';

	interface Props {
		role: MessageRole.USER | MessageRole.ASSISTANT;
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		onRegenerate?: () => void;
		onContinue?: () => void;
		showRawOutputSwitch?: boolean;
		rawOutputEnabled?: boolean;
		onRawOutputToggle?: (enabled: boolean) => void;
	}

	let {
		actionsPosition,
		justify,
		onContinue,
		onRawOutputToggle,
		onRegenerate,
		rawOutputEnabled = false,
		role,
		showRawOutputSwitch = false
	}: Props = $props();

	const messageActions = getChatMessageActionsContext();
	const editCtx = getChatMessageEditContext();

	let showForkDialog = $state(false);
	let forkName = $state('');
	let forkIncludeAttachments = $state(true);

	function handleConfirmDelete() {
		messageActions.confirmDelete();
		messageActions.setShowDeleteDialog(false);
	}

	function handleOpenForkDialog() {
		const conv = conversationsStore.activeConversation;

		forkName = `${conv?.name ?? '当前会话'}（分支）`;
		forkIncludeAttachments = true;
		showForkDialog = true;
	}

	function handleConfirmFork() {
		messageActions.forkConversation?.({
			includeAttachments: forkIncludeAttachments,
			name: forkName.trim()
		});
		showForkDialog = false;
	}
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-between">
	<div
		class="{actionsPosition === 'left'
			? 'left-0'
			: 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
	>
		{#if messageActions.siblingInfo && messageActions.siblingInfo.totalSiblings > 1}
			<ChatMessageActionIconsBranchingControls />
		{/if}

		<div
			class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
		>
			<ActionIcon icon={Copy} onclick={messageActions.copy} tooltip="复制" />

			<ActionIcon icon={Edit} onclick={editCtx.startEdit} tooltip="编辑" />

			{#if role === MessageRole.ASSISTANT && onRegenerate}
				<ActionIcon icon={RefreshCw} onclick={() => onRegenerate()} tooltip="重新生成" />
			{/if}

			{#if role === MessageRole.ASSISTANT && onContinue}
				<ActionIcon icon={ArrowRight} onclick={onContinue} tooltip="继续生成" />
			{/if}

			{#if messageActions.forkConversation}
				<ActionIcon icon={GitBranch} onclick={handleOpenForkDialog} tooltip="从此消息创建分支对话" />
			{/if}

			<ActionIcon icon={Trash2} onclick={messageActions.requestDelete} tooltip="删除" />
		</div>
	</div>

	{#if showRawOutputSwitch}
		<div class="flex items-center gap-2">
			<span class="text-xs text-muted-foreground">显示原始文本</span>

			<Switch
				checked={rawOutputEnabled}
				onCheckedChange={(checked) => onRawOutputToggle?.(checked)}
			/>
		</div>
	{/if}
</div>

<DialogConfirmation
	cancelText="取消"
	confirmText={messageActions.deletionInfo && messageActions.deletionInfo.totalCount > 1
		? `删除 ${messageActions.deletionInfo.totalCount} 条消息`
		: '删除'}
	description={messageActions.deletionInfo && messageActions.deletionInfo.totalCount > 1
		? `此操作将删除 ${messageActions.deletionInfo.totalCount} 条消息（包含 ${messageActions.deletionInfo.userMessages} 条用户提问与 ${messageActions.deletionInfo.assistantMessages} 条助手回复）。该分支下的所有后续消息将被永久移除，且无法撤销。`
		: '确定要删除这条消息吗？此操作无法撤销。'}
	icon={Trash2}
	onCancel={() => messageActions.setShowDeleteDialog(false)}
	onConfirm={handleConfirmDelete}
	open={messageActions.showDeleteDialog}
	title="删除消息"
	variant="destructive"
/>

<DialogConfirmation
	bind:open={showForkDialog}
	cancelText="取消"
	confirmText="创建分支"
	description="从当前消息节点分叉创建一个新的独立对话。"
	icon={GitBranch}
	onCancel={() => (showForkDialog = false)}
	onConfirm={handleConfirmFork}
	title="创建分支对话"
>
	<div class="flex flex-col gap-4 py-2">
		<div class="flex flex-col gap-2">
			<Label for="fork-name">分支标题</Label>

			<Input
				bind:value={forkName}
				class="text-foreground"
				id="fork-name"
				placeholder="请输入新分支对话名称"
				type="text"
			/>
		</div>

		<div class="flex items-center gap-2">
			<Checkbox
				checked={forkIncludeAttachments}
				id="fork-attachments"
				onCheckedChange={(checked) => {
					forkIncludeAttachments = checked === true;
				}}
			/>

			<Label class="cursor-pointer text-sm font-normal" for="fork-attachments">
				包含所有附件
			</Label>
		</div>
	</div>
</DialogConfirmation>
