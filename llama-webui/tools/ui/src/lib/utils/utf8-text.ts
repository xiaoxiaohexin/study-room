/**
 * Repair text where llama.cpp streamed a multi-byte UTF-8 character as
 * individual bytes mapped to Latin-1 code points (e.g. "æ¶©" for "波").
 *
 * Bytes that never form a complete character are dropped: they are token
 * fragments, not text. Text that is already valid UTF-8 passes through
 * unchanged because CJK and emoji are above U+00FF.
 */
export function sanitizeUtf8Text(text: string): string {
	let out = '';
	let pending: number[] = [];

	const flush = (): void => {
		if (pending.length === 0) return;

		const bytes = new Uint8Array(pending);

		pending = [];

		// Keep the longest valid prefix; an unfinished character at the tail is noise.
		for (let end = bytes.length; end > 0; end--) {
			try {
				out += new TextDecoder('utf-8', { fatal: true }).decode(bytes.subarray(0, end));

				return;
			} catch {
				// try a shorter prefix
			}
		}
	};

	for (const ch of text) {
		const codePoint = ch.codePointAt(0) ?? 0;

		if (codePoint >= 0x80 && codePoint <= 0xff) {
			pending.push(codePoint);

			continue;
		}

		flush();
		out += ch;
	}

	flush();

	return out;
}
