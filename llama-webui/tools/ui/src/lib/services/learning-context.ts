/**
 * Learning-context helpers: single source of truth for the active
 * course / RAG namespace / user identity used by learning APIs.
 *
 * Keeping these in one module prevents the canvas, chat selection toolbar and
 * knowledge-base dialog from drifting back to hard-coded `default` values.
 */

import { conversationsStore } from '$lib/stores';
import { getAuthHeaders } from '$lib/utils/api-headers';

export const DEFAULT_USER_ID = 'default_user';
export const DEFAULT_COURSE_ID = 'default';
export const DEFAULT_RAG_NAMESPACE = 'default';

/** WebUI identity (matching the backend llama_compat default). */
export function getActiveUserId(): string {
	return DEFAULT_USER_ID;
}

/** Course bound to the active conversation, falling back to the virtual default course. */
export function getActiveCourseId(): string {
	const courseId = conversationsStore.preferences?.getCourseId?.();
	return (courseId && courseId.trim()) || DEFAULT_COURSE_ID;
}

/** RAG namespace bound to the active conversation, falling back to `default`. */
export function getActiveRagNamespace(): string {
	const namespace = conversationsStore.preferences?.getRagNamespace?.();
	return (namespace && namespace.trim()) || DEFAULT_RAG_NAMESPACE;
}

/** Headers carrying the learning scope for backend requests. */
export function learningHeaders(): Record<string, string> {
	return {
		...getAuthHeaders(),
		'X-User-Id': getActiveUserId(),
		'X-RAG-Namespace': getActiveRagNamespace(),
		'X-Course-Id': getActiveCourseId()
	};
}

/** JSON headers plus learning scope headers. */
export function learningJsonHeaders(): Record<string, string> {
	return {
		'Content-Type': 'application/json',
		...learningHeaders()
	};
}
