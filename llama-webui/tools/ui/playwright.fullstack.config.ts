import { defineConfig, devices } from '@playwright/test';

/**
 * Full-stack E2E: real SvelteKit dev server + real FastAPI backend in offline mode.
 * Uses the system Chrome channel so no Playwright browser download is required.
 *
 * Ports are isolated from the developer's own uvicorn (8000) / dev server (5173):
 * backend 8010, frontend 8182.
 */
export default defineConfig({
	expect: { timeout: 15000 },
	fullyParallel: false,
	workers: 1,
	retries: 0,
	reporter: 'line',
	testDir: 'tests/e2e-fullstack',
	testMatch: ['**/*.e2e.ts'],
	timeout: 240000,
	use: {
		baseURL: 'http://127.0.0.1:8182',
		viewport: { width: 1500, height: 950 },
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	projects: [
		{
			name: 'chrome',
			use: {
				...devices['Desktop Chrome'],
				channel: 'chrome',
				viewport: { width: 1500, height: 950 }
			}
		}
	],
	webServer: [
		{
			command: '.venv\\Scripts\\python.exe tests/e2e_server.py',
			cwd: '../../..',
			url: 'http://127.0.0.1:8010/health',
			reuseExistingServer: false,
			timeout: 120000,
			stdout: 'ignore',
			stderr: 'pipe'
		},
		{
			// 本地 fixture 网页服务器：供「联网导入」E2E 抓取（后端已允许回环地址）
			command:
				'.venv\\Scripts\\python.exe -m http.server 8011 --bind 127.0.0.1 --directory llama-webui/tools/ui/tests/e2e-fullstack/fixtures',
			cwd: '../../..',
			url: 'http://127.0.0.1:8011/web_page.html',
			reuseExistingServer: false,
			timeout: 30000,
			stdout: 'ignore',
			stderr: 'pipe'
		},
		{
			command: 'npm run dev -- --port 8182 --host 127.0.0.1 --mode e2e',
			url: 'http://127.0.0.1:8182',
			reuseExistingServer: false,
			timeout: 180000,
			stdout: 'ignore',
			stderr: 'pipe'
		}
	]
});
