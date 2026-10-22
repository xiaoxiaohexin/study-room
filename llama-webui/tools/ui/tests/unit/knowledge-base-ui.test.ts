import { describe, expect, it } from 'vitest';
import {
	PRESET_KNOWLEDGE_BASES,
	getKnowledgeBaseBadgeLabel
} from '$lib/constants/knowledge-base.constants';

describe('Knowledge Base UI helper & constants', () => {
	it('defines standard preset knowledge bases', () => {
		const ids = PRESET_KNOWLEDGE_BASES.map((k) => k.id);
		expect(ids).toContain('default');
		expect(ids).toContain('calculus');
		expect(ids).toContain('linear_algebra');
		expect(ids).toContain('physics');
		expect(ids).toContain('cs_network');
	});

	it('returns null badge for default namespace with no courseId', () => {
		expect(getKnowledgeBaseBadgeLabel('default')).toBeNull();
		expect(getKnowledgeBaseBadgeLabel(undefined)).toBeNull();
		expect(getKnowledgeBaseBadgeLabel('')).toBeNull();
	});

	it('returns compact badge for preset knowledge bases', () => {
		expect(getKnowledgeBaseBadgeLabel('calculus')).toBe('高数');
		expect(getKnowledgeBaseBadgeLabel('physics')).toBe('物理');
		expect(getKnowledgeBaseBadgeLabel('linear_algebra')).toBe('线代');
		expect(getKnowledgeBaseBadgeLabel('cs_network')).toBe('计网');
	});

	it('returns raw namespace or courseId for custom knowledge bases', () => {
		expect(getKnowledgeBaseBadgeLabel('custom_robotics')).toBe('custom_robotics');
		expect(getKnowledgeBaseBadgeLabel('default', 'MATH_101')).toBe('MATH_101');
	});
});
