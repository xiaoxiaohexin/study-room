/**
 * Knowledge Base (RAG Namespace) configuration and domain presets for the general Agent.
 */

export interface KnowledgeBaseOption {
	id: string; // rag_namespace
	label: string;
	courseId?: string;
	description?: string;
	shortBadge?: string;
}

export const PRESET_KNOWLEDGE_BASES: KnowledgeBaseOption[] = [
	{
		id: 'default',
		label: '默认知识库 (通用)',
		description: '默认通用知识库命名空间',
		shortBadge: '默认'
	},
	{
		id: 'calculus',
		label: '高等数学 (微积分)',
		courseId: 'math_calc',
		description: '极限、微分学、积分学与级数',
		shortBadge: '高数'
	},
	{
		id: 'linear_algebra',
		label: '线性代数',
		courseId: 'math_linalg',
		description: '矩阵论、线性方程组与特征空间',
		shortBadge: '线代'
	},
	{
		id: 'physics',
		label: '大学物理',
		courseId: 'phys_gen',
		description: '经典力学、电磁学与热力学',
		shortBadge: '物理'
	},
	{
		id: 'cs_network',
		label: '计算机网络',
		courseId: 'cs_net',
		description: 'TCP/IP 体系、网络协议与路由',
		shortBadge: '计网'
	}
];

export function getKnowledgeBaseBadgeLabel(ragNamespace?: string, courseId?: string): string | null {
	if (!ragNamespace || ragNamespace === 'default') {
		if (courseId) return courseId;
		return null;
	}

	const found = PRESET_KNOWLEDGE_BASES.find((item) => item.id === ragNamespace);
	if (found?.shortBadge) {
		return found.shortBadge;
	}

	return ragNamespace;
}
