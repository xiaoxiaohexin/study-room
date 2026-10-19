import { FILE_TYPE_ICONS } from '$lib/constants';
import { AttachmentAction, AttachmentItemEnabledWhen, AttachmentMenuItemId } from '$lib/enums';
import type { AttachmentMenuItem } from '$lib/types';

/**
 * File attachment menu items shown in both the desktop dropdown and mobile sheet.
 * The "Tools" submenu is handled separately by each component.
 */
export const ATTACHMENT_FILE_ITEMS: AttachmentMenuItem[] = [
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'images-button',
		disabledTooltip: '图片处理需要支持视觉（Vision）的模型',
		enabledWhen: AttachmentItemEnabledWhen.HAS_VISION_MODALITY,
		icon: FILE_TYPE_ICONS.image,
		id: AttachmentMenuItemId.IMAGES,
		label: '图片'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'audio-button',
		disabledTooltip: '音频处理需要支持音频（Audio）的模型',
		enabledWhen: AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY,
		icon: FILE_TYPE_ICONS.audio,
		id: AttachmentMenuItemId.AUDIO,
		label: '音频文件'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		class: 'video-button',
		disabledTooltip: '视频处理需要支持视频（Video）的模型',
		enabledWhen: AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY,
		icon: FILE_TYPE_ICONS.video,
		id: AttachmentMenuItemId.VIDEO,
		label: '视频文件'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		icon: FILE_TYPE_ICONS.text,
		id: AttachmentMenuItemId.TEXT,
		label: '文本文件'
	},
	{
		action: AttachmentAction.FILE_UPLOAD,
		disabledTooltip: 'PDF 将转换为文本处理，纯图片扫描版 PDF 可能无法正常识别。',
		enabledWhen: AttachmentItemEnabledWhen.ALWAYS,
		hasEnabledTooltip: true,
		icon: FILE_TYPE_ICONS.pdf,
		id: AttachmentMenuItemId.PDF,
		label: 'PDF 文档'
	}
];

export const ATTACHMENT_TOOLTIP_TEXT = '添加文件、提示词、工具或 MCP 服务';

/** Text shared by the add menu's two surfaces: the desktop dropdown and the mobile drawer. */
export const ATTACHMENT_MENU_TEXT = {
	ADD_FILES: '添加文件',
	ADD_TO_CHAT: '添加到对话',
	DESCRIPTION: '添加附件、系统提示词或选择模型可调用的工具',
	SYSTEM_MESSAGE: '系统提示词',
	TOOLS: '工具'
} as const;

/** Modality labels behind the file-upload row's enabled-when flags, both surfaces. */
export const ATTACHMENT_MODALITY_TEXT: Record<AttachmentItemEnabledWhen, string> = {
	[AttachmentItemEnabledWhen.ALWAYS]: '',
	[AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]: '音频',
	[AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]: '视频',
	[AttachmentItemEnabledWhen.HAS_VISION_MODALITY]: '视觉'
};
