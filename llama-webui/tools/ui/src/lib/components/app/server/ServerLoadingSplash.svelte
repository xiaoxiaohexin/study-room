<script lang="ts">
	import { Server } from '@lucide/svelte';
	import { ServerStatus } from '$lib/components/app';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		message?: string;
	}

	let { class: className = '', message = '正在初始化连接后端服务…' }: Props = $props();
</script>

<div class="flex h-full items-center justify-center {className}">
	<div class="text-center">
		<div in:fade={{ duration: 300 }} class="mb-4">
			<div class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
				<Server class="h-8 w-8 animate-pulse text-muted-foreground" />
			</div>

			<h2 class="mb-2 text-xl font-semibold">正在连接服务端</h2>

			<p class="text-sm text-muted-foreground">
				{message}
			</p>
		</div>

		<div class="mt-4">
			<ServerStatus class="justify-center" />
		</div>
	</div>
</div>
