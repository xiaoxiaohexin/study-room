// Registry of server and browser tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new tool, add an entry to TOOL_UI. To give a
// tool a custom title or body renderer, add a dedicated component under
// ChatMessageToolCall/ and route it in ChatMessageToolCallBlock.svelte
// (see ChatMessageToolCallBlockGetDatetime and
// ChatMessageToolCallBlockSearchResults for prior art).

import {
	Braces,
	Clock,
	Eye,
	FilePen,
	FilePlus,
	FileSearch,
	FileText,
	Info,
	SearchCode,
	Terminal
} from '@lucide/svelte';
import { BuiltInTool, ToolSource } from '$lib/enums';
import type { ToolUiEntry } from '$lib/types';

export const TOOL_UI: Readonly<Record<BuiltInTool, ToolUiEntry>> = {
	[BuiltInTool.BROWSER_GET_DATETIME]: {
		icon: Clock,
		label: '获取当前时间',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.BROWSER_READ_MEDIA]: { icon: Eye, label: '读取媒体文件', source: ToolSource.BROWSER },
	[BuiltInTool.BROWSER_RUN_JAVASCRIPT]: {
		icon: Braces,
		label: '运行 JavaScript',
		source: ToolSource.BROWSER
	},
	[BuiltInTool.SERVER_EDIT_FILE]: { icon: FilePen, label: '编辑文件', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_EXEC_SHELL_COMMAND]: {
		icon: Terminal,
		label: '执行终端命令',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_FILE_GLOB_SEARCH]: {
		icon: FileSearch,
		label: '搜索文件',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_GET_INFO]: { icon: Info, label: '运行时信息', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_GREP_SEARCH]: {
		icon: SearchCode,
		label: '搜索文件内容',
		source: ToolSource.SERVER
	},
	[BuiltInTool.SERVER_READ_FILE]: { icon: FileText, label: '读取文件', source: ToolSource.SERVER },
	[BuiltInTool.SERVER_WRITE_FILE]: {
		icon: FilePlus,
		label: '写入文件',
		source: ToolSource.SERVER
	}
} as const;
