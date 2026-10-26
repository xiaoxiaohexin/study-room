<script lang="ts">
	import { BookOpen, Check } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Drawer from '$lib/components/ui/drawer';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Input } from '$lib/components/ui/input';
	import { ICON_CLASS_DEFAULT } from '$lib/constants';
	import {
		PRESET_KNOWLEDGE_BASES,
		type KnowledgeBaseOption
	} from '$lib/constants/knowledge-base.constants';
	import { conversationsStore, deviceStore } from '$lib/stores';

	interface Props {
		disabled?: boolean;
	}

	let { disabled = false }: Props = $props();

	let isOpen = $state(false);
	let customNamespaceInput = $state('');

	let isMobile = $derived(deviceStore.isMobile);

	let activeConv = $derived(conversationsStore.activeConversation);
	let currentNamespace = $derived(
		activeConv
			? (activeConv.ragNamespace || 'default')
			: (conversationsStore.preferences.pendingRagNamespace || 'default')
	);
	let currentCourseId = $derived(
		activeConv ? activeConv.courseId : (conversationsStore.preferences.pendingCourseId ?? undefined)
	);

	let currentPreset = $derived(
		PRESET_KNOWLEDGE_BASES.find((item) => item.id === currentNamespace)
	);

	let displayLabel = $derived.by(() => {
		if (currentPreset) return currentPreset.label;
		return `知识库: ${currentNamespace}`;
	});

	let isDefault = $derived(currentNamespace === 'default');

	async function selectKnowledgeBase(option: KnowledgeBaseOption) {
		const newNamespace = option.id;
		const newCourseId = option.courseId ?? null;

		await conversationsStore.preferences.setRagNamespace(newNamespace, newCourseId);

		isOpen = false;
	}

	async function handleCustomSubmit() {
		const trimmed = customNamespaceInput.trim();
		if (!trimmed) return;

		await selectKnowledgeBase({
			id: trimmed,
			label: `自定义 (${trimmed})`,
			courseId: undefined
		});
		customNamespaceInput = '';
	}
</script>

{#snippet triggerContent()}
	<span class="flex items-center gap-1.5 {isDefault ? 'text-muted-foreground' : 'text-foreground font-medium'}">
		<BookOpen class="size-3.5 max-md:size-4 shrink-0" />
		<span class="max-w-[120px] truncate text-xs">{displayLabel}</span>
	</span>
{/snippet}

{#snippet optionItem(option: KnowledgeBaseOption)}
	{@const isSelected = currentNamespace === option.id}
	<button
		aria-pressed={isSelected}
		class="flex min-h-9 w-full cursor-pointer items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors hover:bg-accent"
		onclick={() => selectKnowledgeBase(option)}
		type="button"
	>
		{#if isSelected}
			<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
		{:else}
			<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
		{/if}

		<div class="min-w-0 flex-1">
			<div class="truncate font-medium">{option.label}</div>
			{#if option.description}
				<div class="truncate text-[11px] text-muted-foreground">{option.description}</div>
			{/if}
		</div>
	</button>
{/snippet}

{#if isMobile}
	<Button
		aria-label="选择知识库"
		class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! px-1.75! py-1 text-xs max-md:h-8 hover:bg-transparent!"
		{disabled}
		onclick={() => (isOpen = true)}
		variant="ghost"
	>
		{@render triggerContent()}
	</Button>

	<Drawer.Root bind:open={isOpen}>
		<Drawer.Content>
			<Drawer.Header>
				<Drawer.Title>选择知识库</Drawer.Title>
			</Drawer.Header>

			<div class="flex flex-col gap-1 px-3 pb-6">
				{#each PRESET_KNOWLEDGE_BASES as option (option.id)}
					{@render optionItem(option)}
				{/each}

				{#if !currentPreset && currentNamespace !== 'default'}
					{@render optionItem({
						id: currentNamespace,
						label: `当前自定义: ${currentNamespace}`,
						courseId: currentCourseId
					})}
				{/if}

				<div class="mt-3 flex items-center gap-2 border-t pt-3">
					<Input
						bind:value={customNamespaceInput}
						class="h-8 text-xs"
						onkeydown={(e: KeyboardEvent) => {
							if (e.key === 'Enter') {
								e.preventDefault();
								void handleCustomSubmit();
							}
						}}
						placeholder="输入自定义知识库命名空间..."
					/>
					<Button
						class="h-8 shrink-0 text-xs"
						disabled={!customNamespaceInput.trim()}
						onclick={handleCustomSubmit}
						size="sm"
					>
						绑定
					</Button>
				</div>
			</div>
		</Drawer.Content>
	</Drawer.Root>
{:else}
	<DropdownMenu.Root bind:open={isOpen}>
		<DropdownMenu.Trigger>
			{#snippet child({ props })}
				<Button
					{...props}
					aria-label="选择知识库"
					class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! px-1.75! py-1 text-xs hover:bg-transparent!"
					{disabled}
					variant="ghost"
				>
					{@render triggerContent()}
				</Button>
			{/snippet}
		</DropdownMenu.Trigger>

		<DropdownMenu.Content align="end" class="min-w-64 p-1.5">
			<div class="px-2 py-1 text-xs font-semibold text-muted-foreground select-none">
				知识域与知识库
			</div>

			{#each PRESET_KNOWLEDGE_BASES as option (option.id)}
				{@const isSelected = currentNamespace === option.id}
				<DropdownMenu.Item
					class="gap-2.5 cursor-pointer py-1.5"
					onSelect={() => selectKnowledgeBase(option)}
				>
					{#if isSelected}
						<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
					{:else}
						<div class="{ICON_CLASS_DEFAULT} shrink-0"></div>
					{/if}

					<div class="min-w-0 flex-1">
						<div class="truncate text-xs font-medium">{option.label}</div>
						{#if option.description}
							<div class="truncate text-[10px] text-muted-foreground">{option.description}</div>
						{/if}
					</div>
				</DropdownMenu.Item>
			{/each}

			{#if !currentPreset && currentNamespace !== 'default'}
				<DropdownMenu.Item
					class="gap-2.5 cursor-pointer py-1.5"
					onSelect={() => {}}
				>
					<Check class="{ICON_CLASS_DEFAULT} shrink-0 text-foreground" />
					<div class="min-w-0 flex-1">
						<div class="truncate text-xs font-medium">自定义: {currentNamespace}</div>
						<div class="truncate text-[10px] text-muted-foreground">当前绑定的命名空间</div>
					</div>
				</DropdownMenu.Item>
			{/if}

			<div class="mt-2 border-t pt-2 px-1">
				<div class="flex items-center gap-1.5">
					<Input
						bind:value={customNamespaceInput}
						class="h-7 text-xs"
						onkeydown={(e: KeyboardEvent) => {
							if (e.key === 'Enter') {
								e.preventDefault();
								void handleCustomSubmit();
							}
						}}
						placeholder="自定义命名空间..."
					/>
					<Button
						class="h-7 shrink-0 px-2 text-xs"
						disabled={!customNamespaceInput.trim()}
						onclick={handleCustomSubmit}
						size="sm"
						variant="secondary"
					>
						确定
					</Button>
				</div>
			</div>
		</DropdownMenu.Content>
	</DropdownMenu.Root>
{/if}
