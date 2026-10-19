/** Labels of the models manager table sections. */

import { ModelsTableGroupKind } from '$lib/enums';

export const MODELS_TABLE_GROUP_LABELS: Record<ModelsTableGroupKind, string> = {
	[ModelsTableGroupKind.DOWNLOADING]: '下载中',
	[ModelsTableGroupKind.FAVORITES]: '收藏模型',
	[ModelsTableGroupKind.HIDDEN]: '已隐藏模型',
	[ModelsTableGroupKind.LOADED]: '已加载模型',
	[ModelsTableGroupKind.LOCAL]: '本地模型'
};

/**
 * Sticky offset of a group heading under its section header: one section-header
 * row (`py-2` over `text-[13px]` line height) minus the header's bottom border,
 * so the two stacked sticky rows do not show a gap or overlap by a pixel.
 */
export const MODELS_TABLE_GROUP_STICKY_OFFSET = 'top: calc(2.25rem - 1px)';
