<script lang="ts">
	import { Download, Pin, PinOff, Trash2, X } from '@lucide/svelte';
	import { ActionIcon, DialogConfirmation } from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { TooltipSide } from '$lib/enums';

	interface Props {
		class?: string;
		selectedCount: number;
		visibleCount: number;
		allVisibleSelected: boolean;
		someVisibleSelected: boolean;
		someSelectedPinned: boolean;
		pinStateIsMixed: boolean;
		onSelectAllToggle: () => void;
		onBulkPinToggle: () => void;
		onBulkExport: () => void;
		onBulkDelete: () => void;
		onClose: () => void;
	}

	let {
		allVisibleSelected,
		class: className = '',
		onBulkDelete,
		onBulkExport,
		onBulkPinToggle,
		onClose,
		onSelectAllToggle,
		pinStateIsMixed,
		selectedCount,
		someSelectedPinned,
		someVisibleSelected,
		visibleCount
	}: Props = $props();

	let showDeleteDialog = $state(false);

	function handleDeleteClick() {
		showDeleteDialog = true;
	}

	function handleDeleteConfirm() {
		showDeleteDialog = false;
		onBulkDelete();
	}

	function handleDeleteCancel() {
		showDeleteDialog = false;
	}

	const hasSelection = $derived(selectedCount > 0);
	const isMasterChecked = $derived(allVisibleSelected);
	const isMasterIndeterminate = $derived(!allVisibleSelected && someVisibleSelected);

	const pinTooltip = $derived(
		hasSelection
			? pinStateIsMixed
				? '选中项置顶状态不一致，无法批量切换'
				: someSelectedPinned
					? selectedCount === 1
						? '取消置顶'
						: '全部取消置顶'
					: selectedCount === 1
						? '置顶'
						: '全部置顶'
			: '置顶'
	);

	const pinDisabled = $derived(!hasSelection || pinStateIsMixed);
</script>

<div
	aria-label="批量操作选中对话"
	class="flex items-center gap-1.5 rounded-xl border border-border/50 bg-background/50 px-2 py-1.5 shadow-sm backdrop-blur-xl {className}"
	role="toolbar"
>
	<label class="flex min-w-0 cursor-pointer items-center gap-2">
		<Checkbox
			aria-label={isMasterChecked ? '取消全选' : '全选'}
			checked={isMasterChecked}
			indeterminate={isMasterIndeterminate}
			onCheckedChange={onSelectAllToggle}
		/>

		<span class="truncate text-xs font-medium text-muted-foreground">
			已选 {selectedCount} / {visibleCount}
		</span>
	</label>

	<div class="ml-auto flex items-center gap-0.75">
		<ActionIcon
			ariaLabel={pinTooltip}
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {pinDisabled
				? 'cursor-not-allowed'
				: ''} {!pinDisabled ? 'opacity-100' : 'opacity-40'}"
			disabled={pinDisabled}
			icon={someSelectedPinned ? PinOff : Pin}
			iconSize="h-3.5 w-3.5"
			onclick={onBulkPinToggle}
			size="sm"
			tooltip={pinTooltip}
			tooltipSide={TooltipSide.TOP}
		/>

		<ActionIcon
			ariaLabel="导出选中对话"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {hasSelection
				? 'opacity-100'
				: 'opacity-40'}"
			disabled={!hasSelection}
			icon={Download}
			iconSize="h-3.5 w-3.5"
			onclick={onBulkExport}
			size="sm"
			tooltip="导出选中对话"
			tooltipSide={TooltipSide.TOP}
		/>

		<ActionIcon
			ariaLabel="删除选中对话"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-destructive/10! dark:hover:bg-destructive/20! disabled:hover:bg-transparent {hasSelection
				? 'opacity-100'
				: 'opacity-40'}"
			disabled={!hasSelection}
			icon={Trash2}
			iconSize="h-3.5 w-3.5 text-destructive"
			onclick={handleDeleteClick}
			size="sm"
			tooltip="删除选中对话"
			tooltipSide={TooltipSide.TOP}
		/>

		<div aria-hidden="true" class="mx-1 h-4 w-px bg-border"></div>

		<ActionIcon
			ariaLabel="退出批量选择"
			class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent!"
			icon={X}
			iconSize="h-3.5 w-3.5"
			onclick={onClose}
			size="sm"
			tooltip="退出批量选择"
			tooltipSide={TooltipSide.TOP}
		/>
	</div>
</div>

<DialogConfirmation
	bind:open={showDeleteDialog}
	cancelText="取消"
	confirmText={selectedCount === 1 ? '删除' : `删除 ${selectedCount} 个对话`}
	description="此操作无法撤销。选中的 {selectedCount} 个对话及其所有消息（含分支）将被永久删除。"
	icon={Trash2}
	onCancel={handleDeleteCancel}
	onConfirm={handleDeleteConfirm}
	title="删除 {selectedCount} 个对话"
	variant="destructive"
/>
