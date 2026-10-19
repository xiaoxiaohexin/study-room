import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const REASONING_EFFORT_LABELS: Record<string, string> = {
	[ReasoningEffort.DEFAULT]: '默认',
	[ReasoningEffort.HIGH]: '高',
	[ReasoningEffort.LOW]: '低',
	[ReasoningEffort.MAX]: '最高',
	[ReasoningEffort.MEDIUM]: '中',
	[ReasoningEffort.OFF]: '关闭'
};

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ label: '默认', value: ReasoningEffort.DEFAULT },
	{ label: '关闭', value: ReasoningEffort.OFF },
	{ label: '低', value: ReasoningEffort.LOW },
	{ label: '中', value: ReasoningEffort.MEDIUM },
	{ label: '高', value: ReasoningEffort.HIGH },
	{ hasInfo: true, label: '最高', value: ReasoningEffort.MAX }
];

/**
 * Reasoning effort to token budget mapping.
 * Maps the ReasoningEffort enum values to concrete token counts for the server.
 */
export const REASONING_EFFORT_TOKENS: Record<string, number> = {
	[ReasoningEffort.HIGH]: 8192,
	[ReasoningEffort.LOW]: 512,
	[ReasoningEffort.MAX]: -1, // unlimited
	[ReasoningEffort.MEDIUM]: 2048
};
