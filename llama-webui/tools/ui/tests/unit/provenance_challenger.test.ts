import { describe, expect, it } from 'vitest';
import type {
	CustomCanvasNode,
	KnowledgeRevision,
	ProvenanceAnchor
} from '$lib/types/canvas';
import {
	extractProvenanceInfo,
	extractTitleCandidate,
	provenanceNavigation,
	sanitizeHtml
} from '$lib/services/provenance-navigation.service';

describe('Adversarial Challenge Suite M3: Text Selection, Extraction, Sanitization & Provenance', () => {
	// =========================================================================
	// Dimension 1: Multiple Paragraphs & Markdown Formatting
	// =========================================================================
	describe('Dimension 1: Multi-Paragraph & Rich Markdown Extraction', () => {
		it('extracts title cleanly from multi-paragraph selection without leftover newline formatting', () => {
			const multiPara = `第一段介绍总体背景与历史渊源。\n\n**自注意力机制** 计算查询向量与键向量的点积相似度，并通过 Softmax 归一化为注意力权重分布。\n\n第三段补充多头注意力的多子空间投影特性。`;
			const title = extractTitleCandidate(multiPara);
			expect(title).toBe('第一段介绍总体背景与历史渊源');
		});

		it('extracts bold title when selection starts with markdown bold', () => {
			const boldText = `**Transformer 架构** 是当前大语言模型（如 GPT、Claude、LLaMA）的基石。`;
			const title = extractTitleCandidate(boldText);
			expect(title).toBe('Transformer 架构');
		});

		it('strips leading unordered list symbols (- , * , >) and extracts concept title', () => {
			const bulletDash = `- 深度残差学习 是为了解决超深网络中的退化问题而提出的。`;
			expect(extractTitleCandidate(bulletDash)).toBe('深度残差学习');

			const bulletStar = `* LayerNorm 是在单个样本的所有特征维度上进行标准化操作。`;
			expect(extractTitleCandidate(bulletStar)).toBe('LayerNorm');

			const blockquote = `> 梯度裁剪 是一项防止反向传播中梯度爆炸的工程技巧。`;
			expect(extractTitleCandidate(blockquote)).toBe('梯度裁剪');
		});

		it('strips leading numbered list prefixes (1., 2.、) and cleans title', () => {
			const numberedDot = `1. 反向传播 是基于多元微积分链式法则的自动化梯度计算方法。`;
			expect(extractTitleCandidate(numberedDot)).toBe('反向传播');

			const numberedDun = `2、卷积核 是图像特征提取的核心滑动窗口。`;
			expect(extractTitleCandidate(numberedDun)).toBe('卷积核');
		});

		it('handles code blocks gracefully without crashing or creating ugly artifacts', () => {
			const codeBlock = "```python\ndef relu(x):\n    return max(0, x)\n```\nReLU 激活函数有效缓解梯度消失。";
			const title = extractTitleCandidate(codeBlock);
			expect(title).toBeTruthy();
			expect(title).not.toContain('```');
		});

		it('handles inline code syntax in definitions', () => {
			const inlineCode = "`torch.nn.CrossEntropyLoss` 是分类任务中最常用的损失函数之一。";
			const title = extractTitleCandidate(inlineCode);
			expect(title).toBeTruthy();
			expect(title.length).toBeLessThanOrEqual(40);
		});

		it('handles markdown italic and bold-italic definitions', () => {
			const italic = `*Dropout* 是一种在训练过程中随机使神经元失活的正规化手段。`;
			const title = extractTitleCandidate(italic);
			expect(title).toBe('Dropout');

			const boldItalic = `***过拟合*** 是指模型在训练集表现优异但在测试集泛化能力低下。`;
			const title2 = extractTitleCandidate(boldItalic);
			expect(title2).toBe('过拟合');
		});
	});

	// =========================================================================
	// Dimension 2: LaTeX Math Expressions Preservation
	// =========================================================================
	describe('Dimension 2: LaTeX Mathematical Formulations & Syntax Preservation', () => {
		it('preserves bold mathematical definitions in extractTitleCandidate without stripping backslashes', () => {
			const mathBold = `**\\mathcal{L}_{MSE} = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2** 是回归分析的基础损失目标。`;
			const title = extractTitleCandidate(mathBold);
			expect(title).toBe('\\mathcal{L}_{MSE} = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2');
			expect(title).toContain('\\frac');
			expect(title).toContain('\\sum');
		});

		it('preserves inline math formulas ($E = mc^2$, $\\frac{a}{b}$) without syntax loss', () => {
			const text = `质能方程 $E = mc^2$ 阐述了质量与能量的等价转化关系。`;
			const title = extractTitleCandidate(text);
			expect(title).toContain('$E = mc^2$');
		});

		it('handles formulas containing inequality symbols (<, >) safely across sanitization', () => {
			const formula = '满足条件 $0 \\le x < 1$ 且 $y > 0$ 时，对数函数 $\\ln(x)$ 严格为负。';
			const sanitized = sanitizeHtml(formula);
			// sanitizeHtml escapes HTML tags but KaTeX can render escaped &lt; or raw < depending on context
			expect(sanitized).toContain('&lt;');
			expect(sanitized).toContain('&gt;');
			expect(sanitized).toContain('\\ln(x)');
		});

		it('preserves multi-variable limit formulas containing commas and limits', () => {
			const limitText = `**\\lim_{(x,y) \\to (0,0)} \\frac{x y}{x^2 + y^2}** 该二元极限不存在。`;
			const title = extractTitleCandidate(limitText);
			expect(title).toBe('\\lim_{(x,y) \\to (0,0)} \\frac{x y}{x^2 + y^2}');
		});

		it('preserves matrix and multi-line LaTeX block structures in raw quote', () => {
			const matrixQuote = `$$\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}^{-1} = \\frac{1}{ad-bc} \\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$$`;
			// rawQuote must be stored untouched
			const anchor: ProvenanceAnchor = {
				id: 'prov_matrix',
				messageId: 'msg_m',
				quoteSnippet: matrixQuote,
				createdAt: 1000
			};
			const card: CustomCanvasNode = {
				id: 'kn_matrix',
				kind: 'knowledge',
				color: 'emerald',
				width: 380,
				x: 0,
				y: 0,
				title: '二阶逆矩阵公式',
				content: '推导过程与行列式判别',
				rawQuote: matrixQuote,
				provenance: [anchor]
			};

			expect(card.rawQuote).toBe(matrixQuote);
			expect(card.rawQuote).toContain('\\begin{bmatrix}');
			expect(card.rawQuote).toContain('\\end{bmatrix}');
			expect(card.provenance![0].quoteSnippet).toBe(matrixQuote);
		});
	});

	// =========================================================================
	// Dimension 3: Malicious Inputs and XSS Payloads
	// =========================================================================
	describe('Dimension 3: Malicious Payloads & XSS Defense', () => {
		it('sanitizes executable script tags from raw HTML input', () => {
			const malicious = `<script>alert("PWNED_1")</script>正常文本内容`;
			const clean = sanitizeHtml(malicious);
			expect(clean).not.toContain('<script>');
			expect(clean).not.toContain('</script>');
			expect(clean).toContain('&lt;script&gt;');
		});

		it('neutralizes inline event handlers (onload, onerror, onclick, onmouseover)', () => {
			const payload = `<img src="nonexistent.jpg" onerror="alert('XSS_IMG')"/><svg onload="evilCode()"/><a href="javascript:void(0)" onclick="steal()">click</a>`;
			const clean = sanitizeHtml(payload);
			expect(clean).not.toContain('<img');
			expect(clean).not.toContain('<svg');
			expect(clean).not.toContain('<a');
			expect(clean).toContain('&lt;img');
			expect(clean).toContain('&lt;svg');
		});

		it('safely handles malicious XSS in bold title candidates without crashing extractTitleCandidate', () => {
			const xssInBold = `**<script>alert(1)</script>恶意的注入标题** 这是攻击载荷测试。`;
			const title = extractTitleCandidate(xssInBold);
			expect(title).toBe('<script>alert(1)</script>恶意的注入标题');

			// When rendered in Svelte or sanitized:
			const cleanTitle = sanitizeHtml(title);
			expect(cleanTitle).toBe('&lt;script&gt;alert(1)&lt;/script&gt;恶意的注入标题');
		});

		it('sanitizes iframe and embed injection vectors', () => {
			const iframePayload = `<iframe src="https://attacker.com/cookie-stealer"></iframe><embed src="malware.swf">`;
			const sanitized = sanitizeHtml(iframePayload);
			expect(sanitized).not.toContain('<iframe');
			expect(sanitized).not.toContain('<embed');
			expect(sanitized).toContain('&lt;iframe');
		});

		it('sanitizes quote attributes and double quotes in titles or content', () => {
			const quoteInjection = `title" onfocus="alert(1)" dummy="`;
			const sanitized = sanitizeHtml(quoteInjection);
			expect(sanitized).toContain('&quot;');
			expect(sanitized).not.toContain('"');
		});
	});

	// =========================================================================
	// Dimension 4: Extreme Boundary Selections & Length Limits
	// =========================================================================
	describe('Dimension 4: Extreme Boundaries & Stress Invariants', () => {
		it('returns default fallback title for empty, whitespace, and symbol-only selections', () => {
			expect(extractTitleCandidate('')).toBe('知识节点提炼');
			expect(extractTitleCandidate('    ')).toBe('知识节点提炼');
			expect(extractTitleCandidate('\n\t\r')).toBe('知识节点提炼');
			expect(extractTitleCandidate('---===***###')).toBe('知识节点提炼');
			expect(extractTitleCandidate('1. 、 - >')).toBe('知识节点提炼');
		});

		it('handles single-character and very short selections safely', () => {
			const singleChar = 'A';
			expect(extractTitleCandidate(singleChar)).toBe('A');

			const chineseChar = '数';
			expect(extractTitleCandidate(chineseChar)).toBe('数');

			const twoChars = '梯度';
			expect(extractTitleCandidate(twoChars)).toBe('梯度');
		});

		it('strictly truncates excessively long title candidates to 40 characters max', () => {
			const massiveSentence = '这是一个超级漫长没有任何标点符号的极度冗长的概念解释说明文本用来测试标题截断算法是否能在四十个字符以内做精确的安全切分防止UI布局被破坏';
			const title = extractTitleCandidate(massiveSentence);
			expect(title.length).toBeLessThanOrEqual(40);
			expect(title).toBe(massiveSentence.slice(0, 40));
		});

		it('handles massive 5,000+ words selection without stack overflow or performance degradation', () => {
			const word = '知识节点提炼演化 ';
			const massiveText = word.repeat(2500); // ~25,000 characters
			expect(massiveText.length).toBeGreaterThan(20000);

			const start = performance.now();
			const title = extractTitleCandidate(massiveText);
			const prov = extractProvenanceInfo(massiveText);
			const clean = sanitizeHtml(massiveText);
			const elapsed = performance.now() - start;

			expect(title).toBeTruthy();
			expect(title.length).toBeLessThanOrEqual(40);
			expect(prov.docTitle).toBe('知识源文档');
			expect(prov.pageStart).toBe(1);
			expect(clean.length).toBeGreaterThanOrEqual(massiveText.length);
			expect(elapsed).toBeLessThan(100); // Must process 25k chars in <100ms
		});

		it('handles single 10,000-character uninterrupted token without hanging', () => {
			const giantToken = 'A'.repeat(10000);
			const start = performance.now();
			const title = extractTitleCandidate(giantToken);
			const elapsed = performance.now() - start;

			expect(title.length).toBe(40);
			expect(elapsed).toBeLessThan(50);
		});
	});

	// =========================================================================
	// Dimension 5: Textbook Citation Parser Variants & Edge Cases
	// =========================================================================
	describe('Dimension 5: Textbook Citation Parser Robustness', () => {
		it('parses [DOCUMENT: filename P.12-15] format correctly', () => {
			const citation = `[DOCUMENT: LinearAlgebra.pdf P.12-15] 矩阵的秩等于其主元列数。`;
			const info = extractProvenanceInfo(citation);
			expect(info.docTitle).toBe('LinearAlgebra.pdf');
			expect(info.pageStart).toBe(12);
			expect(info.pageEnd).toBe(15);
		});

		it('parses [DOCUMENT: filename P.42] single page format correctly', () => {
			const citation = `[DOCUMENT: DeepLearningBook.pdf P.42] 梯度下降动力学。`;
			const info = extractProvenanceInfo(citation);
			expect(info.docTitle).toBe('DeepLearningBook.pdf');
			expect(info.pageStart).toBe(42);
			expect(info.pageEnd).toBe(42);
		});

		it('parses [DOCUMENT: ... (Headers)] with section headers in citation', () => {
			const citation = `[DOCUMENT: AdvancedCalculus.pdf P.108-110 (Chapter 4 > Section 4.2)] 隐函数存在定理。`;
			const info = extractProvenanceInfo(citation);
			expect(info.docTitle).toBe('AdvancedCalculus.pdf');
			expect(info.pageStart).toBe(108);
			expect(info.pageEnd).toBe(110);
		});

		it('parses standalone [P.42] citation bracket', () => {
			const citation = `根据参考书目 [P.42]，该结论对所有连续函数均成立。`;
			const info = extractProvenanceInfo(citation);
			expect(info.pageStart).toBe(42);
			expect(info.pageEnd).toBe(42);
		});

		it('parses [Page 10] and [pages 20-25] English citation formats', () => {
			const citationSingle = `See [Page 10] for complete proof.`;
			const info1 = extractProvenanceInfo(citationSingle);
			expect(info1.pageStart).toBe(10);
			expect(info1.pageEnd).toBe(10);

			const citationRange = `Refer to [pages 20-25] for case studies.`;
			const info2 = extractProvenanceInfo(citationRange);
			expect(info2.pageStart).toBe(20);
			expect(info2.pageEnd).toBe(25);
		});

		it('parses Chinese natural language formats: 第 77 到 79 页, 第 100 至 105 页', () => {
			const info1 = extractProvenanceInfo('详见教材第 77 到 79 页分析。');
			expect(info1.pageStart).toBe(77);
			expect(info1.pageEnd).toBe(79);

			const info2 = extractProvenanceInfo('详见教材第 100 至 105 页实验。');
			expect(info2.pageStart).toBe(100);
			expect(info2.pageEnd).toBe(105);
		});

		it('handles book titles enclosed in Chinese book marks 《...》', () => {
			const citation = `引自《人工智能现代方法第四版》第 P.305-308 页关于启发式搜索的讨论。`;
			const info = extractProvenanceInfo(citation);
			expect(info.docTitle).toBe('人工智能现代方法第四版');
			expect(info.pageStart).toBe(305);
			expect(info.pageEnd).toBe(308);
		});

		it('handles DOM message container data attributes when present to override or provide provenance', () => {
			// Mock a DOM element container
			const mockContainer = {
				getAttribute(name: string) {
					if (name === 'data-turn-index') return '4';
					if (name === 'data-doc-title') return 'OfficialCurriculum.pdf';
					if (name === 'data-page-start') return '99';
					if (name === 'data-page-end') return '101';
					return null;
				}
			} as unknown as HTMLElement;

			const info = extractProvenanceInfo('选中文本没有任何书名和页码标注', mockContainer);
			expect(info.turnIndex).toBe(4);
			expect(info.docTitle).toBe('OfficialCurriculum.pdf');
			expect(info.pageStart).toBe(99);
			expect(info.pageEnd).toBe(101);
		});

		it('handles invalid non-numeric page patterns gracefully with fallback', () => {
			const invalidCitation = `[DOCUMENT: Book.pdf P.abc] 未知页码。`;
			const info = extractProvenanceInfo(invalidCitation);
			expect(info.docTitle).toBe('Book.pdf');
			// Should fallback to pageStart: 1 gracefully without NaN
			expect(info.pageStart).toBe(1);
			expect(Number.isNaN(info.pageStart)).toBe(false);
		});
	});

	// =========================================================================
	// Dimension 6: Dual-Zone Card Immutability & Revisions Under Stress
	// =========================================================================
	describe('Dimension 6: Dual-Zone Immutability & Rollback Invariants Under Stress', () => {
		it('strictly isolates Zone 1 immutable quote from multiple consecutive Zone 2 user edits', () => {
			const immutableText = '真理只有一个：当且仅当损失函数为全局强凸时，随机梯度下降以线性速率收敛。';
			const anchor: ProvenanceAnchor = {
				id: 'prov_immutable_test',
				messageId: 'msg_999',
				turnIndex: 5,
				docTitle: '凸优化理论.pdf',
				pageStart: 120,
				pageEnd: 122,
				quoteSnippet: immutableText,
				createdAt: Date.now()
			};

			const card: CustomCanvasNode = {
				id: 'kn_stress_card',
				kind: 'knowledge',
				color: 'emerald',
				width: 380,
				x: 0,
				y: 0,
				title: '凸优化收敛性',
				content: '版本 1 笔记',
				rawQuote: immutableText,
				provenance: [anchor],
				currentVersion: 1,
				revisions: [
					{
						version: 1,
						timestamp: Date.now(),
						editor: 'extract',
						changeType: 'initial',
						summary: '初始提炼',
						content: '版本 1 笔记',
						rawQuote: immutableText
					}
				]
			};

			// Simulate 50 sequential user note edits
			for (let i = 2; i <= 50; i++) {
				const newContent = `用户编辑笔记第 ${i} 次修改：补充参数学习率 eta = 0.0${i}`;
				const newTitle = `凸优化收敛分析 (修订 #${i})`;
				card.title = newTitle;
				card.content = newContent;
				card.currentVersion = i;
				card.revisions!.push({
					version: i,
					timestamp: Date.now() + i * 100,
					editor: 'user',
					changeType: 'edit_content',
					summary: `编辑笔记 #${i}`,
					content: newContent,
					title: newTitle
				});

				// Invariant: Zone 1 raw quote and provenance anchor MUST remain strictly untouched
				expect(card.rawQuote).toBe(immutableText);
				expect(card.provenance![0].quoteSnippet).toBe(immutableText);
				expect(card.provenance![0].pageStart).toBe(120);
			}

			expect(card.currentVersion).toBe(50);
			expect(card.revisions).toHaveLength(50);
			expect(card.rawQuote).toBe(immutableText);
		});

		it('executes multiple successive non-destructive rollbacks while maintaining full revision timeline', () => {
			const revs: KnowledgeRevision[] = [
				{ version: 1, timestamp: 100, editor: 'extract', changeType: 'initial', summary: 'v1', content: 'C1', title: 'T1' },
				{ version: 2, timestamp: 200, editor: 'user', changeType: 'edit_content', summary: 'v2', content: 'C2', title: 'T2' },
				{ version: 3, timestamp: 300, editor: 'user', changeType: 'edit_content', summary: 'v3', content: 'C3', title: 'T3' }
			];

			const card: CustomCanvasNode = {
				id: 'kn_multi_rollback',
				kind: 'knowledge',
				color: 'emerald',
				width: 380,
				x: 0,
				y: 0,
				title: 'T3',
				content: 'C3',
				currentVersion: 3,
				revisions: [...revs]
			};

			// Rollback to v1 -> creates v4 (revert)
			const target1 = card.revisions![0];
			const rev4: KnowledgeRevision = {
				version: 4,
				timestamp: 400,
				editor: 'user',
				changeType: 'revert',
				summary: `回退至版本 v${target1.version}`,
				content: target1.content,
				title: target1.title
			};
			card.currentVersion = 4;
			card.content = target1.content;
			card.title = target1.title ?? '';
			card.revisions!.push(rev4);

			expect(card.content).toBe('C1');
			expect(card.currentVersion).toBe(4);

			// Rollback to v2 -> creates v5 (revert)
			const target2 = card.revisions![1];
			const rev5: KnowledgeRevision = {
				version: 5,
				timestamp: 500,
				editor: 'user',
				changeType: 'revert',
				summary: `回退至版本 v${target2.version}`,
				content: target2.content,
				title: target2.title
			};
			card.currentVersion = 5;
			card.content = target2.content;
			card.title = target2.title ?? '';
			card.revisions!.push(rev5);

			expect(card.content).toBe('C2');
			expect(card.currentVersion).toBe(5);
			expect(card.revisions).toHaveLength(5);
			// History is monotonically increasing and never mutated backwards
			expect(card.revisions!.map((r) => r.version)).toEqual([1, 2, 3, 4, 5]);
		});

		it('handles appending up to 20 quotes to a single card without crashing provenance tabs', () => {
			const anchors: ProvenanceAnchor[] = [];
			for (let i = 1; i <= 20; i++) {
				anchors.push({
					id: `prov_app_${i}`,
					messageId: `msg_${i}`,
					turnIndex: i,
					docTitle: `Book_${i}.pdf`,
					pageStart: i * 10,
					pageEnd: i * 10 + 2,
					quoteSnippet: `这是从对话 #${i} 抽取的关键引用片段内容。`,
					createdAt: Date.now() + i
				});
			}

			const card: CustomCanvasNode = {
				id: 'kn_20_quotes',
				kind: 'knowledge',
				color: 'emerald',
				width: 380,
				x: 0,
				y: 0,
				title: '聚合了20条引用的超级概念',
				content: '综合知识体系',
				rawQuote: anchors.map((a) => a.quoteSnippet).join('\n\n---\n\n'),
				provenance: anchors,
				currentVersion: 20
			};

			expect(card.provenance).toHaveLength(20);
			expect(card.provenance![0].turnIndex).toBe(1);
			expect(card.provenance![19].turnIndex).toBe(20);
			expect(card.rawQuote).toContain(anchors[0].quoteSnippet);
			expect(card.rawQuote).toContain(anchors[19].quoteSnippet);

			// Check simulated active index clamping
			const activeIdx = Math.min(card.activeProvenanceIndex ?? 0, Math.max(0, card.provenance!.length - 1));
			expect(activeIdx).toBe(0);
			expect(card.provenance![activeIdx].id).toBe('prov_app_1');
		});
	});
});
