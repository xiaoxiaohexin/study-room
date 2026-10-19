<script lang="ts">
	import { RotateCcw } from '@lucide/svelte';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Button } from '$lib/components/ui/button';
	import { settingsStore } from '$lib/stores';

	interface Props {
		onReset?: () => void;
		onSave?: () => void;
	}

	let { onReset, onSave }: Props = $props();

	let showResetDialog = $state(false);

	function handleResetClick() {
		showResetDialog = true;
	}

	function handleConfirmReset() {
		settingsStore.forceSyncWithServerDefaults();
		onReset?.();

		showResetDialog = false;
	}

	function handleSave() {
		onSave?.();
	}
</script>

<div class="sticky bottom-0 mx-auto mt-4 flex w-full justify-between pb-4 md:pb-0">
	<div class="flex gap-2">
		<Button onclick={handleResetClick} variant="outline">
			<RotateCcw class="h-3 w-3" />

			恢复默认设置
		</Button>
	</div>

	<Button onclick={handleSave}>保存设置</Button>
</div>

<AlertDialog.Root bind:open={showResetDialog}>
	<AlertDialog.Content>
		<AlertDialog.Header>
			<AlertDialog.Title>恢复默认设置</AlertDialog.Title>

			<AlertDialog.Description>
				确定要将所有设置恢复为默认值吗？此操作会将所有采样与配置参数重置为服务端 /props 接口提供的默认值，并清除你的自定义修改。
			</AlertDialog.Description>
		</AlertDialog.Header>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>取消</AlertDialog.Cancel>

			<AlertDialog.Action onclick={handleConfirmReset}>确认恢复默认</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
