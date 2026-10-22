/**
 * Real conversation turn numbering for message anchors.
 *
 * `buildSiblingInfoMap` exposes a branch-sibling index, which is always 0 in a
 * linear chat; provenance badges must use the user-turn ordinal instead.
 */

export interface TurnIndexEntry {
	id: string;
	role: string;
}

const USER_ROLE = 'user';
const ASSISTANT_ROLE = 'assistant';

/** Map message id -> 1-based user turn number (assistant replies share the turn). */
export function buildTurnIndexMap(entries: TurnIndexEntry[]): Map<string, number> {
	const map = new Map<string, number>();
	let turn = 0;
	for (const entry of entries) {
		if (entry.role === USER_ROLE) turn += 1;
		if (entry.role === USER_ROLE || entry.role === ASSISTANT_ROLE) {
			map.set(entry.id, Math.max(turn, 1));
		}
	}
	return map;
}
