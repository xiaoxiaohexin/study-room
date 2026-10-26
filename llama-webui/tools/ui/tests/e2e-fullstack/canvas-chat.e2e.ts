import fs from 'node:fs';
import path from 'node:path';
import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const PDF_FIXTURE = path.resolve(
	process.cwd(),
	'tests/stories/fixtures/assets/example.pdf'
);

test.describe.configure({ mode: 'serial' });

function chatInput(page: Page) {
	return page.locator('.conversation-chat-form').locator('[role="textbox"], textarea').first();
}

async function readChatInput(page: Page): Promise<string> {
	return chatInput(page).evaluate((el) => {
		if (el instanceof HTMLTextAreaElement) return el.value;
		return el.textContent ?? '';
	});
}

async function openChat(page: Page): Promise<void> {
	await page.addInitScript(() => {
		localStorage.setItem('llama-preferred-view-mode', 'split');
	});
	await page.goto('/', { waitUntil: 'domcontentloaded' });
	// First Vite dev page load compiles the whole SvelteKit app; allow a cold start.
	await expect(chatInput(page)).toBeVisible({ timeout: 180000 });
}

async function sendMessage(page: Page, text: string): Promise<void> {
	const assistantMessages = page.locator('.chat-messages [data-message-role="assistant"]');
	const before = await assistantMessages.count();
	const input = chatInput(page);
	// WebUI keeps the composer disabled until the model/server handshake finishes.
	await expect(input).toBeEnabled({ timeout: 120000 });
	await input.click();
	await page.keyboard.insertText(text);
	await page.keyboard.press('Enter');
	await expect(assistantMessages.nth(before)).toBeVisible({ timeout: 90000 });
	await expect(assistantMessages.nth(before)).toContainText('FakeLLM', { timeout: 90000 });
}

async function selectAssistantText(page: Page, charCount = 12): Promise<void> {
	await page.evaluate((count) => {
		const container = document.querySelector('.chat-messages [data-message-role="assistant"]');
		if (!container) throw new Error('assistant message not found');
		const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
		let target: Text | null = null;
		while (walker.nextNode()) {
			const node = walker.currentNode as Text;
			if ((node.textContent ?? '').trim().length >= count) {
				target = node;
				break;
			}
		}
		if (!target) throw new Error('no selectable text node in assistant message');
		const range = document.createRange();
		range.setStart(target, 0);
		range.setEnd(target, Math.min(count, target.textContent?.length ?? 0));
		const selection = window.getSelection();
		if (!selection) throw new Error('no selection');
		selection.removeAllRanges();
		selection.addRange(range);
		document
			.querySelector('.chat-messages')
			?.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
	}, charCount);
	await expect(page.getByRole('button', { name: /提取为知识节点/ })).toBeVisible({
		timeout: 15000
	});
}

async function createKnowledgeViaSelection(page: Page) {
	await selectAssistantText(page);
	await page.getByRole('button', { name: /提取为知识节点/ }).click();
	const card = page.locator('[data-node-kind="knowledge"]').last();
	await expect(card).toBeVisible({ timeout: 30000 });
	await expect(card.getByRole('button', { name: /对话 #/ })).toBeVisible({ timeout: 10000 });
	return card;
}

/** Find the canvas card id whose title input equals `title` (null when absent). */
async function findCardNodeIdByTitle(page: Page, title: string): Promise<string | null> {
	return page.evaluate((name) => {
		const cards = Array.from(
			document.querySelectorAll<HTMLElement>('[data-node-kind="knowledge"]')
		);
		for (const card of cards) {
			const input = card.querySelector('input') as HTMLInputElement | null;
			if (input?.value === name) return card.getAttribute('data-node-id');
		}
		return null;
	}, title);
}

async function jsClick(locator: import('@playwright/test').Locator): Promise<void> {
	// Canvas cards can overlap each other; dispatch a DOM click so pointer-event
	// interception by sibling canvas cards does not block the test.
	await locator.evaluate((el) => (el as HTMLElement).click());
}

test('画布 ↔ 对话：划词建卡、锚点跳转、分支追问与分支切换', async ({ page }) => {
	await openChat(page);

	// --- 1. 两轮对话，为分支切换准备兄弟节点 ---
	await sendMessage(page, '请简单介绍一下牛顿第二定律');
	await sendMessage(page, '再用一句话总结一下');

	// --- 2. 划词建卡：真实落库 + 真实轮次锚点（无伪造"参考教材 P.1"） ---
	const card = await createKnowledgeViaSelection(page);
	await expect(page.getByText('参考教材 P.1')).toHaveCount(0);
	await expect(card.getByRole('button', { name: /对话 #/ })).toBeVisible();

	// 后端确实写入了知识节点（默认用户 / 默认课程）
	const kpResponse = await page.request.get('/api/courses/default/knowledge', {
		headers: { 'X-User-Id': 'default_user' }
	});
	expect(kpResponse.ok()).toBeTruthy();
	const kpData = await kpResponse.json();
	expect(kpData.count).toBeGreaterThanOrEqual(1);

	// --- 3. 卡片 → 对话锚点：点击徽章后消息被高亮 ---
	await jsClick(card.getByRole('button', { name: /对话 #/ }));
	await expect
		.poll(
			() =>
				page.evaluate(() => {
					const messages = document.querySelectorAll('.chat-messages [data-message-id]');
					return Array.from(messages).some((el) =>
						el.classList.contains('pulse-provenance-highlight')
					);
				}),
			{ timeout: 10000 }
		)
		.toBe(true);

	// --- 4. 从对话节点发起分支追问，画布出现分支指示与切换器 ---
	const turnOne = page.locator('[data-node-id^="turn-"]').first();
	await jsClick(turnOne.getByRole('button', { name: /分支追问/ }));
	await expect.poll(() => readChatInput(page), { timeout: 15000 }).toContain('分支追问');
	await chatInput(page).click();
	await page.keyboard.press('End');
	await page.keyboard.insertText('继续深入讲解');
	await page.keyboard.press('Enter');

	const assistantMessages = page.locator('.chat-messages [data-message-role="assistant"]');
	await expect(assistantMessages.first()).toBeVisible({ timeout: 30000 });
	await expect(page.getByLabel('切换到上一个对话分支').first()).toBeVisible({
		timeout: 30000
	});

	// 切换到另一个分支，原分支消息仍在
	await jsClick(page.getByLabel('切换到上一个对话分支').first());
	await expect(page.getByText(/分支 1\/2/).first()).toBeVisible({ timeout: 15000 });
	await expect(assistantMessages.first()).toBeVisible();

	// 切回新分支
	await jsClick(page.getByLabel('切换到下一个对话分支').first());
	await expect(page.getByText(/分支 2\/2/).first()).toBeVisible({ timeout: 15000 });
});

test('真实教材页码溯源：后端文档注册 → 画布补齐 → 渲染真实 PDF 页', async ({ page }) => {
	// 1) 通过真实后端上传 PDF 并登记文档
	const pdfBytes = fs.readFileSync(PDF_FIXTURE);
	const ingest = await page.request.post('/api/rag/ingest', {
		multipart: {
			namespace: 'default',
			file: {
				name: 'example.pdf',
				mimeType: 'application/pdf',
				buffer: pdfBytes
			}
		}
	});
	expect(ingest.ok()).toBeTruthy();
	const ingestData = await ingest.json();
	expect(ingestData.chunks).toBeGreaterThan(0);

	// 2) 创建带结构化文档锚点（doc_id + 物理页）的知识点
	const kpName = `USENIX 示例论文 ${Date.now()}`;
	const kpCreate = await page.request.post('/api/courses/default/knowledge', {
		headers: { 'X-User-Id': 'default_user' },
		data: {
			name: kpName,
			summary: '示例论文页级溯源',
			content: '讲义笔记',
			raw_quote: 'USENIX Example Paper',
			provenance: {
				message_id: 'seed-message',
				turn_index: 1,
				doc_id: ingestData.doc_id,
				doc_title: 'example.pdf',
				page_start: 1,
				page_end: 1,
				quote_snippet: 'USENIX Example Paper'
			}
		}
	});
	expect(kpCreate.ok()).toBeTruthy();

	// 3) 进入对话触发画布后端补齐，应出现该知识卡片
	await openChat(page);
	await sendMessage(page, '测试教材溯源卡片补齐');
	await expect
		.poll(() => findCardNodeIdByTitle(page, kpName), { timeout: 30000 })
		.not.toBeNull();
	const nodeId = await findCardNodeIdByTitle(page, kpName);
	expect(nodeId).toBeTruthy();
	const targetCard = page.locator(`[data-node-id="${nodeId}"]`);
	await expect(targetCard).toBeVisible();

	// 4) 点击教材徽章 → 模态框渲染真实 PDF 页面（pdf.js），而不是占位假段落
	const badge = targetCard.getByRole('button', { name: /example\.pdf\s*P\.1/ });
	await expect(badge).toBeVisible({ timeout: 15000 });
	await badge.click();

	const dialog = page.getByRole('dialog', { name: '文档溯源预览' });
	await expect(dialog).toBeVisible({ timeout: 10000 });
	await expect(dialog.locator('img')).toBeVisible({ timeout: 30000 });
	await expect(dialog.getByText(/pdf\.js 渲染/)).toBeVisible({ timeout: 10000 });
	await expect(page.getByText('在相应章节中，本节重点阐述了')).toHaveCount(0);
});

test('联网导入 → 临时知识库 → 提纯候选 → 晋升标准知识库 (WebStaging)', async ({ page }) => {
	await page.addInitScript(() => {
		localStorage.setItem('llama-preferred-view-mode', 'split');
	});
	await page.goto('/', { waitUntil: 'domcontentloaded' });

	// 打开知识库管理对话框
	await expect(page.getByRole('button', { name: '导入知识库' })).toBeVisible({
		timeout: 180000
	});
	const browserStatus = await page.evaluate(async () => {
		const response = await fetch('/api/web/status');
		return { ok: response.ok, body: await response.json() };
	});
	expect(browserStatus.ok).toBe(true);
	expect(browserStatus.body.enabled).toBe(true);
	await page.getByRole('button', { name: '导入知识库' }).click();
	const dialog = page.getByRole('dialog');
	await expect(dialog).toBeVisible({ timeout: 15000 });
	await expect(dialog.getByText('联网功能未开启')).toHaveCount(0);

	// 联网导入 tab：抓取本地 fixture 网页（后端 WEB_ALLOW_LOOPBACK=true）
	await dialog.getByRole('button', { name: /联网导入/ }).click();
	await dialog.getByPlaceholder(/粘贴网页 URL/).fill('http://127.0.0.1:8011/web_page.html');
	await dialog.getByRole('button', { name: '抓取预览' }).click();
	await expect(dialog.getByText('E2E 光合作用讲义').first()).toBeVisible({
		timeout: 30000
	});

	// 存入临时知识库（标准库此时不应有该文档）
	await dialog.getByRole('button', { name: '存入临时知识库' }).click();
	await expect(dialog.getByText('E2E 光合作用讲义').first()).toBeVisible({
		timeout: 30000
	});
	const beforeDocs = await page.request.get('/api/rag/documents?namespace=default');
	const beforeBody = await beforeDocs.json();
	expect(
		beforeBody.documents.some((doc: { filename?: string }) =>
			String(doc.filename ?? '').includes('E2E 光合作用讲义')
		)
	).toBe(false);

	// 提纯 → 生成候选 → 勾选晋升
	await dialog.getByRole('button', { name: '提纯' }).first().click();
	await expect(dialog.getByText(/待入库知识候选/)).toBeVisible({ timeout: 30000 });
	const candidateCheckbox = dialog.locator('input[type="checkbox"]').first();
	await expect(candidateCheckbox).toBeVisible({ timeout: 30000 });
	await candidateCheckbox.check();
	await dialog.getByRole('button', { name: /晋升到正式库/ }).click();
	await expect(page.getByText(/已晋升/)).toBeVisible({ timeout: 30000 });

	// 后端标准知识库确实出现了该网页快照文档
	const afterDocs = await page.request.get('/api/rag/documents?namespace=default');
	const afterBody = await afterDocs.json();
	expect(
		afterBody.documents.some((doc: { filename?: string }) =>
			String(doc.filename ?? '').includes('E2E 光合作用讲义')
		)
	).toBe(true);
});
