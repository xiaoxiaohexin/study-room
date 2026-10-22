import { describe, expect, it } from 'vitest';
import { buildTurnIndexMap } from '$lib/utils/turn-index';

describe('buildTurnIndexMap', () => {
	it('numbers user turns and assigns assistant replies to the same turn', () => {
		const map = buildTurnIndexMap([
			{ id: 'sys', role: 'system' },
			{ id: 'u1', role: 'user' },
			{ id: 'a1', role: 'assistant' },
			{ id: 'u2', role: 'user' },
			{ id: 'a2', role: 'assistant' },
			{ id: 'u3', role: 'user' }
		]);
		expect(map.get('u1')).toBe(1);
		expect(map.get('a1')).toBe(1);
		expect(map.get('u2')).toBe(2);
		expect(map.get('a2')).toBe(2);
		expect(map.get('u3')).toBe(3);
		expect(map.has('sys')).toBe(false);
	});

	it('ignores tool messages and never returns 0', () => {
		const map = buildTurnIndexMap([
			{ id: 'a0', role: 'assistant' },
			{ id: 't1', role: 'tool' }
		]);
		expect(map.get('a0')).toBe(1);
		expect(map.has('t1')).toBe(false);
	});
});
