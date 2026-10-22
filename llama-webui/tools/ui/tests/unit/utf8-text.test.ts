import { sanitizeUtf8Text } from '$lib/utils/utf8-text';
import { describe, expect, it } from 'vitest';

describe('sanitizeUtf8Text', () => {
	it('leaves valid Chinese and ASCII untouched', () => {
		expect(sanitizeUtf8Text('用户想要知道图片中角色的名字。')).toBe('用户想要知道图片中角色的名字。');
		expect(sanitizeUtf8Text('Thinking Process: 1 + 1 = 2')).toBe('Thinking Process: 1 + 1 = 2');
		expect(sanitizeUtf8Text('emoji 🙂 stays')).toBe('emoji 🙂 stays');
	});

	it('reassembles UTF-8 bytes that arrived as Latin-1 code points', () => {
		// "你好" is E4 BD A0 E5 A5 BD; read as Latin-1 the A0 byte becomes NBSP.
		expect(sanitizeUtf8Text('ä½ å¥½')).toBe('你好');
	});

	it('keeps the longest valid prefix and drops an unfinished tail', () => {
		expect(sanitizeUtf8Text('æ¶©ª')).toBe('涩');
	});

	it('drops isolated byte fragments instead of rendering mojibake', () => {
		expect(sanitizeUtf8Text('café')).toBe('caf');
		expect(sanitizeUtf8Text('æ')).toBe('');
	});
});
