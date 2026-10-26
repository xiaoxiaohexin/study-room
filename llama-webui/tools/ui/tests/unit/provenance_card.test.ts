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

describe('Feature 10: Dual-Zone Knowledge Canvas Card Schema & Immutability', () => {
	it('defines correct knowledge node schema with kind: knowledge and provenance anchor', () => {
		const anchor: ProvenanceAnchor = {
			id: 'prov_test_1',
			messageId: 'msg_001',
			turnIndex: 2,
			docTitle: '高等数学第七版.pdf',
			pageStart: 42,
			pageEnd: 43,
			quoteSnippet: '在满足0/0未定式条件下，分子分母分别求导后极限仍然相同',
			createdAt: Date.now()
		};

		const initialRevision: KnowledgeRevision = {
			version: 1,
			timestamp: Date.now(),
			editor: 'extract',
			changeType: 'initial',
			summary: '初始从对话 #2 提炼',
			content: '个人理解：求导前务必先验证是否为未定式',
			title: '洛必达法则',
			quoteSnippet: anchor.quoteSnippet,
			rawQuote: anchor.quoteSnippet,
			difficulty: 0.5
		};

		const card: CustomCanvasNode = {
			id: 'kn_101',
			kind: 'knowledge',
			title: '洛必达法则',
			content: '个人理解：求导前务必先验证是否为未定式',
			rawQuote: anchor.quoteSnippet,
			color: 'emerald',
			width: 380,
			x: 400,
			y: 200,
			difficulty: 0.5,
			mastery: 0.3,
			currentVersion: 1,
			provenance: [anchor],
			revisions: [initialRevision]
		};

		expect(card.kind).toBe('knowledge');
		expect(card.color).toBe('emerald');
		expect(card.width).toBe(380);
		expect(card.difficulty).toBe(0.5);
		expect(card.mastery).toBe(0.3);
		expect(card.provenance).toHaveLength(1);
		expect(card.provenance![0].turnIndex).toBe(2);
		expect(card.provenance![0].pageStart).toBe(42);
		expect(card.provenance![0].pageEnd).toBe(43);
		expect(card.revisions).toHaveLength(1);
	});

	it('preserves immutable quote zone untouched when user edits content or title', () => {
		const rawOriginal = '均方误差 (MSE) 作为凸函数，通过梯度下降可保证收敛到全局最优解';
		const anchor: ProvenanceAnchor = {
			id: 'prov_mse_1',
			messageId: 'msg_042',
			turnIndex: 1,
			docTitle: '机器学习导论.pdf',
			pageStart: 88,
			pageEnd: 89,
			quoteSnippet: rawOriginal,
			createdAt: Date.now()
		};

		const card: CustomCanvasNode = {
			id: 'kn_mse',
			kind: 'knowledge',
			title: '凸优化特性',
			content: '初始笔记：Hessian 矩阵半正定',
			rawQuote: rawOriginal,
			color: 'emerald',
			width: 380,
			x: 100,
			y: 100,
			currentVersion: 1,
			provenance: [anchor],
			revisions: [
				{
					version: 1,
					timestamp: Date.now(),
					editor: 'extract',
					changeType: 'initial',
					summary: '初始提炼',
					content: '初始笔记：Hessian 矩阵半正定',
					rawQuote: rawOriginal
				}
			]
		};

		// User edits title and notes
		card.title = 'MSE 凸优化与梯度收敛性';
		card.content = '二次型优化分析：学习率需满足 Lipschitz 连续性条件，否则震荡';

		// Immutable raw quote and provenance must remain strictly untouched
		expect(card.rawQuote).toBe(rawOriginal);
		expect(card.provenance![0].quoteSnippet).toBe(rawOriginal);
		expect(card.content).not.toBe(card.rawQuote);
		expect(card.content).toContain('Lipschitz');
	});

	it('supports multiple provenance anchors for appended references', () => {
		const anchor1: ProvenanceAnchor = {
			id: 'prov_1',
			messageId: 'msg_1',
			turnIndex: 1,
			docTitle: '统计学习方法.pdf',
			pageStart: 12,
			pageEnd: 12,
			quoteSnippet: '感知机模型旨在寻找将正负样本完全分开的超平面',
			createdAt: Date.now()
		};

		const anchor2: ProvenanceAnchor = {
			id: 'prov_2',
			messageId: 'msg_3',
			turnIndex: 3,
			docTitle: '模式识别.pdf',
			pageStart: 50,
			pageEnd: 52,
			quoteSnippet: 'Novikoff 定理保证了线性可分时感知机在有限步内必然收敛',
			createdAt: Date.now()
		};

		const card: CustomCanvasNode = {
			id: 'kn_perceptron',
			kind: 'knowledge',
			title: '感知机与收敛性',
			content: '核心判定准则与界限',
			rawQuote: `${anchor1.quoteSnippet}\n\n---\n\n${anchor2.quoteSnippet}`,
			color: 'emerald',
			width: 380,
			x: 200,
			y: 200,
			currentVersion: 2,
			provenance: [anchor1, anchor2]
		};

		expect(card.provenance).toHaveLength(2);
		expect(card.provenance![0].turnIndex).toBe(1);
		expect(card.provenance![1].turnIndex).toBe(3);
		expect(card.provenance![1].pageStart).toBe(50);
		expect(card.provenance![1].pageEnd).toBe(52);
		expect(card.rawQuote).toContain(anchor1.quoteSnippet);
		expect(card.rawQuote).toContain(anchor2.quoteSnippet);
	});
});

describe('Feature 10: Revision History & Non-Destructive Rollback Snapshots', () => {
	it('records sequential revision snapshots for content changes and append operations', () => {
		const rev1: KnowledgeRevision = {
			version: 1,
			timestamp: 1000,
			editor: 'extract',
			changeType: 'initial',
			summary: '初始提炼',
			content: '反向传播算法推导',
			title: 'Backprop',
			quoteSnippet: '通过链式法则计算梯度'
		};

		const rev2: KnowledgeRevision = {
			version: 2,
			timestamp: 2000,
			editor: 'user',
			changeType: 'edit_content',
			summary: '增加推导公式说明',
			content: '链式法则展开：dL/dw = dL/dy * dy/dz * dz/dw',
			title: 'Backprop 链式法则'
		};

		const rev3: KnowledgeRevision = {
			version: 3,
			timestamp: 3000,
			editor: 'extract',
			changeType: 'append_quote',
			summary: '追加来自对话 #3 的引用',
			content: rev2.content,
			title: rev2.title,
			quoteSnippet: '自动微分库（如 PyTorch Autograd）基于计算图拓扑排序实现'
		};

		const card: CustomCanvasNode = {
			id: 'kn_bp',
			kind: 'knowledge',
			title: rev3.title ?? '反向传播算法',
			content: rev3.content,
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0,
			currentVersion: 3,
			revisions: [rev1, rev2, rev3]
		};

		expect(card.revisions).toHaveLength(3);
		expect(card.currentVersion).toBe(3);
		expect(card.revisions![0].changeType).toBe('initial');
		expect(card.revisions![1].changeType).toBe('edit_content');
		expect(card.revisions![2].changeType).toBe('append_quote');
	});

	it('performs non-destructive rollback by creating a new revert revision snapshot', () => {
		const rev1: KnowledgeRevision = {
			version: 1,
			timestamp: 1000,
			editor: 'extract',
			changeType: 'initial',
			summary: '初始概念',
			content: '稳定的初始笔记内容',
			title: '初始标题'
		};

		const rev2: KnowledgeRevision = {
			version: 2,
			timestamp: 2000,
			editor: 'user',
			changeType: 'edit_content',
			summary: '错误修改',
			content: '包含错误的混乱笔记',
			title: '错误标题'
		};

		const card: CustomCanvasNode = {
			id: 'kn_rollback',
			kind: 'knowledge',
			title: rev2.title ?? '错误标题',
			content: rev2.content,
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0,
			currentVersion: 2,
			revisions: [rev1, rev2]
		};

		// Rollback to version 1
		const targetRev = rev1;
		const nextVersion = card.currentVersion! + 1;
		const revertRev: KnowledgeRevision = {
			version: nextVersion,
			timestamp: 3000,
			editor: 'user',
			changeType: 'revert',
			summary: `回退至版本 v${targetRev.version}`,
			content: targetRev.content,
			title: targetRev.title
		};

		card.title = targetRev.title ?? '';
		card.content = targetRev.content;
		card.currentVersion = nextVersion;
		card.revisions = [...card.revisions!, revertRev];

		expect(card.currentVersion).toBe(3);
		expect(card.content).toBe('稳定的初始笔记内容');
		expect(card.title).toBe('初始标题');
		expect(card.revisions).toHaveLength(3);
		expect(card.revisions![2].changeType).toBe('revert');
		expect(card.revisions![2].summary).toBe('回退至版本 v1');
	});
});

describe('Feature 09: Text Selection Parsing & Candidate Extraction', () => {
	it('extracts concept title candidates using bold syntax, definition statements, and clauses', () => {
		// Rule 1: Markdown bold
		const bold = extractTitleCandidate('**注意力机制 (Attention Mechanism)** 是 Transformer 的核心计算单元');
		expect(bold).toBe('注意力机制 (Attention Mechanism)');

		// Rule 2: Definition statement
		const def1 = extractTitleCandidate('残差连接是指在深层神经网络中跨层直接连通特征的结构');
		expect(def1).toBe('残差连接');

		const def2 = extractTitleCandidate('梯度消失是由于深层导数连乘小于1导致');
		expect(def2).toBe('梯度消失');

		// Rule 3: Short clause
		const clause = extractTitleCandidate('交叉熵损失函数衡量两个概率分布之间的距离。');
		expect(clause).toBe('交叉熵损失函数衡量两个概率分布之间的距离');

		// Fallback
		expect(extractTitleCandidate('')).toBe('知识节点提炼');
		expect(extractTitleCandidate('   ')).toBe('知识节点提炼');
	});

	it('preserves raw LaTeX mathematical notations during title extraction', () => {
		const latexQuote = '**\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1** 是高等数学中最经典的极限之一';
		const extracted = extractTitleCandidate(latexQuote);
		expect(extracted).toContain('\\lim');
		expect(extracted).toContain('\\frac');
	});

	it('extracts textbook docTitle and page range accurately', () => {
		// Format: 《书名.pdf》 + P.42-43
		const res1 = extractProvenanceInfo(
			'详见《高等数学第七版.pdf》第 P.42-43 页关于柯西中值定理的证明'
		);
		expect(res1.docTitle).toBe('高等数学第七版.pdf');
		expect(res1.pageStart).toBe(42);
		expect(res1.pageEnd).toBe(43);

		// Format: standalone .pdf and single page P.108
		const res2 = extractProvenanceInfo(
			'根据 LinearAlgebra.pdf 第 P.108 页特征值与特征向量'
		);
		expect(res2.docTitle).toBe('LinearAlgebra.pdf');
		expect(res2.pageStart).toBe(108);
		expect(res2.pageEnd).toBe(108);

		// Format: 第 77 到 79 页
		const res3 = extractProvenanceInfo(
			'参考教材内容第 77 到 79 页'
		);
		expect(res3.pageStart).toBe(77);
		expect(res3.pageEnd).toBe(79);

		// Fallback
		const res4 = extractProvenanceInfo('这是纯文本问答，没有教材引用');
		expect(res4.docTitle).toBe('知识源文档');
		expect(res4.pageStart).toBe(1);
	});

	it('sanitizes dangerous HTML markup to defend against XSS', () => {
		const malicious = '<script>alert("xss")</script><img src="x" onerror="evil()"/>';
		const clean = sanitizeHtml(malicious);
		expect(clean).not.toContain('<script>');
		expect(clean).toContain('&lt;script&gt;');
		expect(clean).toContain('&lt;img');
	});
});

describe('Feature 11: Provenance Navigation Service', () => {
	it('manages highlighted message and spotlight node states', () => {
		expect(provenanceNavigation.highlightedMessageId).toBeNull();
		expect(provenanceNavigation.spotlightNodeId).toBeNull();

		provenanceNavigation.navigateToChatMessage('msg_test_123');
		expect(provenanceNavigation.highlightedMessageId).toBe('msg_test_123');

		provenanceNavigation.focusCanvasNode('kn_spotlight_456');
		expect(provenanceNavigation.spotlightNodeId).toBe('kn_spotlight_456');
	});

	it('opens and closes textbook viewer with correct navigation target', () => {
		const target = {
			docTitle: '操作系统导论.pdf',
			pageStart: 256,
			pageEnd: 257,
			quoteSnippet: '页表机制与快表 (TLB) 协同完成虚拟地址到物理地址的映射'
		};

		provenanceNavigation.openTextbookPreview(target);
		expect(provenanceNavigation.textbookViewerOpen).toBe(true);
		expect(provenanceNavigation.textbookViewerTarget?.docTitle).toBe('操作系统导论.pdf');
		expect(provenanceNavigation.textbookViewerTarget?.pageStart).toBe(256);

		provenanceNavigation.closeTextbookPreview();
		expect(provenanceNavigation.textbookViewerOpen).toBe(false);
		expect(provenanceNavigation.textbookViewerTarget).toBeNull();
	});

	it('filters associated knowledge nodes for a specific chat message', () => {
		const kn1: CustomCanvasNode = {
			id: 'kn_alpha',
			kind: 'knowledge',
			title: 'Alpha Concept',
			content: 'Notes',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0,
			provenance: [
				{
					id: 'p1',
					messageId: 'target_msg_1',
					quoteSnippet: 'Snippet 1',
					createdAt: 1000
				}
			]
		};

		const kn2: CustomCanvasNode = {
			id: 'kn_beta',
			kind: 'knowledge',
			title: 'Beta Concept',
			content: 'Notes',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0,
			provenance: [
				{
					id: 'p2',
					messageId: 'other_msg_2',
					quoteSnippet: 'Snippet 2',
					createdAt: 2000
				}
			]
		};

		provenanceNavigation.setKnowledgeNodes([kn1, kn2]);

		const found = provenanceNavigation.getKnowledgeNodesForMessage('target_msg_1');
		expect(found).toHaveLength(1);
		expect(found[0].id).toBe('kn_alpha');
		expect(found[0].title).toBe('Alpha Concept');

		const none = provenanceNavigation.getKnowledgeNodesForMessage('non_existent_msg');
		expect(none).toHaveLength(0);
	});
});
