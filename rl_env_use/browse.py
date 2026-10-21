"""Browser layer: headless Chromium in a Modal sandbox (or local, RLENV_LOCAL_BROWSER=1), driven through browser-harness-js
(persistent CDP REPL: raw CDP for reading and navigation, guarded page/locator actions for clicks and typing), plus page
capture and evidence matching. This is what lets environments be built from data that plain HTTP and fetch tools cannot reach."""

import asyncio
import html as htmllib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import cloud, net
from .graders import norm

SEARCH_ENGINES = re.compile(r'^https?://([a-z]+\.)?(google|bing|duckduckgo|yahoo|baidu|yandex|search\.brave)\.[a-z.]+/', re.I)

# Installed once per browser in the REPL. One raw CDP session reads the page and navigates (any origin); a guarded Page,
# re-pinned to the current origin, performs clicks and typing (observe -> revalidate -> act once -> verify).
RL_JS = r"""globalThis.rl = {
  sleep: ms => new Promise(r => setTimeout(r, ms)),
  async init(port) {
    await browser.connect(port ? { port } : {});
    const tabs = await browser.tabs();
    rl.target = tabs[0]?.targetId ?? (await session._call('Target.createTarget', { url: 'about:blank' })).targetId;
    rl.sid = (await session._call('Target.attachToTarget', { targetId: rl.target, flatten: true })).sessionId;
    await cdp(rl.sid, 'Page.enable', {});
    rl.page = null; rl.origin = null;
    return true;
  },
  async ev(expression) {
    const r = await cdp(rl.sid, 'Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result?.value;
  },
  async settle(ms = 20000) {
    const end = Date.now() + ms;
    while (Date.now() < end) {
      try { if (await rl.ev('document.readyState') === 'complete') break; } catch (e) {}
      await rl.sleep(250);
    }
    await rl.sleep(400);
  },
  async where() {
    for (let i = 0; i < 20; i++) {
      try { return { url: await rl.ev('location.href'), title: await rl.ev('document.title') }; } catch (e) { await rl.sleep(250); }
    }
    return { url: '', title: '' };
  },
  async goto(url) {
    const nav = await cdp(rl.sid, 'Page.navigate', { url });
    if (nav.errorText) throw new Error(`${nav.errorText} (${url})`);
    await rl.settle();
    return rl.where();
  },
  async back() {
    const h = await cdp(rl.sid, 'Page.getNavigationHistory', {});
    if (h.currentIndex < 1) throw new Error('no earlier page in this tab');
    await cdp(rl.sid, 'Page.navigateToHistoryEntry', { entryId: h.entries[h.currentIndex - 1].id });
    await rl.settle();
    return rl.where();
  },
  async guarded() {
    const u = new URL((await rl.where()).url);
    if (!/^https?:$/.test(u.protocol)) throw new Error('no web page is open: navigate first');
    if (rl.page && rl.origin === u.origin && !rl.page.disposed) return rl.page;
    rl.page?.dispose();
    rl.page = await browser.attach(rl.target, { allowedOrigins: [u.origin] });
    rl.origin = u.origin;
    return rl.page;
  },
  async snapshot(interactive) {
    return (await (await rl.guarded()).snapshot({ interactive, maxChars: 400000 })).tree;
  },
  async act(ref, op, value) {
    const p = await rl.guarded();
    await rl.ev("document.querySelectorAll('a[target]').forEach(a => a.removeAttribute('target'))");  // links stay in this tab
    const loc = p.locator(ref);
    const r = op === 'click' ? await loc.click() : op === 'fill' ? await loc.fill(value) : op === 'press' ? await loc.press(value) : await loc.selectOption(value);
    await rl.settle(10000);
    return { status: r.status, reason: r.reason, verified: r.verified, ...(await rl.where()) };
  },
};
return true;"""


class BrowserError(RuntimeError):
	pass


def host_allowed(url: str, allowed: list[str] | None) -> bool:
	"""`allowed` entries: 'example.com' (and its subdomains) or '*.example.com'."""
	if not allowed or not url.startswith('http'):
		return True
	h = host(url)
	return any(h == d or h.endswith('.' + d) for d in (a.lower().removeprefix('*.').removeprefix('www.') for a in allowed))


def _free_port() -> int:
	with socket.socket() as s:
		s.bind(('127.0.0.1', 0))
		return s.getsockname()[1]


def _local_chrome() -> str:
	for c in (
		os.getenv('CHROME_PATH'),
		'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		'/Applications/Chromium.app/Contents/MacOS/Chromium',
		*(shutil.which(n) for n in ('chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable')),
	):
		if c and os.path.exists(c):
			return c
	raise BrowserError('RLENV_LOCAL_BROWSER=1 needs Chrome or Chromium (set CHROME_PATH)')


def _local_repl() -> Path:
	cli = shutil.which('browser-harness-js')
	if not cli:
		raise BrowserError('RLENV_LOCAL_BROWSER=1 needs browser-harness-js on PATH (https://github.com/monotykamary/browser-harness-js)')
	return Path(cli).resolve().with_name('repl.ts')


class Browser:
	"""One headless Chromium with its own browser-harness-js REPL. Started lazily, killed in a `finally` by its user:
	in Modal it is a sandbox that also stops itself when idle, and with the whole app when this process ends."""

	def __init__(self, allowed_domains: list[str] | None = None, proxy: str | None = None):
		self.allowed, self.proxy = allowed_domains, proxy or os.getenv('RLENV_PROXY')
		self.live = self.url_ = self.headers = None
		self.procs, self.tmp = [], None

	async def start(self) -> 'Browser':
		if self.url_:
			return self
		if os.getenv('RLENV_LOCAL_BROWSER') == '1':
			cdp_port, repl_port = _free_port(), _free_port()
			self.tmp = tempfile.mkdtemp(prefix='rlenv-chrome-')
			flags = [
				'--headless=new',
				'--disable-gpu',
				'--no-first-run',
				'--no-default-browser-check',
				'--disable-blink-features=AutomationControlled',
				f'--remote-debugging-port={cdp_port}',
				f'--user-data-dir={self.tmp}',
				'--window-size=1280,1600',
			]
			self.procs.append(
				subprocess.Popen(
					[_local_chrome(), *flags, *([f'--proxy-server={self.proxy}'] if self.proxy else []), 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
				)
			)
			self.procs.append(
				subprocess.Popen(['node', str(_local_repl())], env=os.environ | {'CDP_REPL_PORT': str(repl_port)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			)
			self.url_, self.headers = f'http://127.0.0.1:{repl_port}', {}
			for _ in range(100):
				try:
					httpx.get(f'http://127.0.0.1:{cdp_port}/json/version', timeout=1).raise_for_status()
					httpx.get(f'{self.url_}/health', timeout=1).raise_for_status()
					break
				except Exception:
					await asyncio.sleep(0.1)
			port = cdp_port
		else:
			self.live, self.url_, token = await cloud.browser_sandbox(self.proxy)
			self.headers, port = {'Authorization': f'Bearer {token}'}, 9222
		try:
			await self.js(RL_JS)
			await self.call('init', port)
		except BaseException:
			await self.kill()
			raise
		return self

	async def js(self, body: str, timeout: float = 90):
		"""Run JS statements in the REPL; the value of their `return` comes back parsed."""
		code = f'const __v = await (async () => {{ {body}\n}})(); return JSON.stringify({{ v: __v ?? null }});'
		try:
			r = await asyncio.wait_for(net.post(f'{self.url_}/eval', code, self.headers, timeout), timeout + 5)
		except TimeoutError as e:
			raise BrowserError(f'browser call timed out after {timeout:.0f}s') from e
		if r.status_code != 200:
			raise BrowserError(r.text.strip().splitlines()[0][:400] if r.text.strip() else f'HTTP {r.status_code}')
		return json.loads(r.text)['v'] if r.text else None

	async def call(self, fn: str, *args, timeout: float = 90):
		return await self.js(f'return await rl.{fn}(...{json.dumps(list(args))});', timeout)

	async def goto(self, url: str) -> dict:
		if not host_allowed(url, self.allowed):
			raise BrowserError(f'{host(url)} is outside the allowed domains {self.allowed}')
		return await self.call('goto', url)

	async def act(self, ref: str, op: str, value=None) -> dict:
		r = await self.call('act', ref, op, value, timeout=60)
		if not host_allowed(r.get('url', ''), self.allowed):
			await self.call('back')
			raise BrowserError(f'that led to {host(r["url"])}, outside the allowed domains {self.allowed}: went back')
		return r

	async def url(self) -> str:
		return (await self.call('where'))['url']

	async def title(self) -> str:
		return (await self.call('where'))['title']

	async def eval(self, expr: str):
		return await self.call('ev', expr)

	async def text(self) -> str:
		return await self.eval(PAGE_TEXT_JS) or ''

	async def snapshot(self, interactive: bool = True) -> str:
		return await self.call('snapshot', interactive)

	async def kill(self) -> None:
		live, self.live, self.url_ = self.live, None, None
		await cloud.terminate(live)
		for p in self.procs:
			p.kill()
		self.procs = []
		if self.tmp:
			shutil.rmtree(self.tmp, ignore_errors=True)
			self.tmp = None

	async def __aenter__(self):
		return await self.start()

	async def __aexit__(self, *exc):
		await self.kill()


def browser(allowed_domains: list[str] | None = None, headless: bool = True) -> Browser:
	"""A browser for one mission, check or episode (unstarted; `start()` or `async with`)."""
	return Browser(allowed_domains)


# the page as a person reads it, including same-origin frames (document viewers such as EDGAR's inline XBRL render the document in an iframe)
PAGE_TEXT_JS = """(() => { const t = d => { let s = d.body ? d.body.innerText : '';
  for (const f of d.querySelectorAll('iframe, frame')) { try { if (f.contentDocument) s += '\\n' + t(f.contentDocument); } catch (e) {} }
  return s; }; return t(document); })()"""


async def page_text(b: Browser) -> str:
	return await b.text()


def quote_in(quote: str, text: str) -> bool:
	"""Whole-token containment after normalization; one-character values allowed only if numeric. A quote may skip table
	cells with '...': each piece must be verbatim, in order, and within 300 characters of the previous one."""
	parts = [p for p in re.split(r'\s*(?:\.\.\.|\u2026)\s*', quote) if p.strip()]
	t = f' {norm(text)} '
	if len(parts) > 1:
		at = 0
		for i, p in enumerate(parts):
			q = norm(p)
			j = t.find(f' {q} ', at) if q else -1
			if j < 0 or (i and j - at > 300):
				return False
			at = j + len(q) + 1
		return True
	q = norm(quote)
	return (len(q) >= 2 or q.isdigit()) and f' {q} ' in t


# Generic "show me everything" pass, run repeatedly until the page stops growing: cookie walls, collapsed <details>,
# "Load more" / "Show 12 more replies" / "View hidden items" buttons, and lazy-loaded content below the fold.
EXPAND_JS = r"""(() => {
  const vis = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden'; };
  let n = 0;
  for (const b of document.querySelectorAll('button, [role=button], a')) {
    const t = (b.innerText || b.getAttribute('aria-label') || '').trim().toLowerCase();
    if (t.length > 40 || !vis(b) || b.dataset.rlenvClicked) continue;
    const href = b.tagName === 'A' ? (b.getAttribute('href') || '') : '';
    const inPage = !href || href.startsWith('#') || href.startsWith('javascript');
    if (/^(accept( all)?( cookies)?|agree|i agree|got it|allow all|ok, got it)$/.test(t) && inPage) { b.click(); b.dataset.rlenvClicked = 1; n++; continue; }
    if (inPage && /(load|show|view|see|read|expand)( \d+)? (more|all|hidden|earlier|older|previous|replies|comments)|^\d+ (hidden|more) (items|replies|comments)|^more replies|^expand( all)?$|^show more$/.test(t)) { b.click(); b.dataset.rlenvClicked = 1; n++; }
  }
  for (const d of document.querySelectorAll('details:not([open])')) { d.open = true; n++; }
  window.scrollTo(0, document.body.scrollHeight);
  return n;
})()"""


async def capture(b: Browser, url: str, expand: bool = True, rounds: int = 8, settle: float = 1.5) -> dict:
	"""Render `url` in a real browser and return its full text after expanding hidden content (no LLM involved)."""
	where = await b.goto(url)
	await asyncio.sleep(settle)
	text, clicks = await b.text(), 0
	for _ in range(rounds if expand else 0):
		n = await b.eval(EXPAND_JS) or 0
		clicks += n
		await asyncio.sleep(settle)
		new = await b.text()
		if not n and len(new) <= len(text):
			break
		text = max(text, new, key=len)  # virtualized lists can shrink after clicks: keep the fullest view
	where = await b.call('where') or where
	return {'url': url, 'final_url': where['url'], 'title': where['title'], 'text': text, 'expand_clicks': clicks, 'via': 'browser'}


def _visible_text(raw: str) -> str:
	"""What a fetch-to-markdown tool shows: rendered-markup text, <noscript> kept, scripts/styles dropped."""
	raw = re.sub(r'(?is)<(script|style|template)[^>]*>.*?</\1>', ' ', raw)
	raw = re.sub(r'(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr|section|article)>', '\n', raw)
	return re.sub(r'[ \t]+', ' ', htmllib.unescape(re.sub(r'<[^>]+>', ' ', raw))).strip()


def _raw_strings(raw: str) -> str:
	"""Most charitable HTTP view: every string in the HTML incl. embedded <script> JSON (e.g. SSR payloads), unescaped."""
	raw = htmllib.unescape(raw)
	raw = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), raw).replace('\\n', '\n').replace('\\"', '"').replace('\\/', '/')
	return re.sub(r'<[^>]+>', ' ', raw)


BLOCK_MARKERS = re.compile(
	r'(captcha|are you a robot|verify you are human|access denied|enable javascript|checking your browser|just a moment|cf-chl|unusual traffic|sign in to (view|continue)|log in to (view|continue))',
	re.I,
)
TRANSIENT = {None, 408, 425, 429, 500, 502, 503, 504}


async def http_capture(url: str, timeout: float = 20, retries: int = 3) -> dict:
	"""The vanilla baseline: what a fetch tool / plain HTTP crawler sees for the same URL. `raw` is the upper bound."""
	for attempt in range(retries):
		try:
			async with httpx.AsyncClient(
				follow_redirects=True,
				timeout=timeout,
				headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'},
			) as c:
				r = await c.get(url)
		except Exception as e:
			r, err = None, f'{type(e).__name__}: {e}'[:200]
		status = r.status_code if r is not None else None
		if status not in TRANSIENT or attempt == retries - 1:
			break
		await asyncio.sleep(2 * (attempt + 1))
	if r is None:
		return {'url': url, 'status': None, 'text': '', 'raw': '', 'blocked': None, 'error': err, 'via': 'http'}
	html = 'html' in r.headers.get('content-type', '')
	text = _visible_text(r.text) if html else r.text
	# a block marker only counts on a short page (a "CAPTCHA" docs link in a long sidebar is not a wall)
	blocked = status in (401, 403, 429, 503) or len(text.strip()) < 200 or (bool(BLOCK_MARKERS.search(text[:3000])) and len(text) < 2000)
	return {'url': url, 'status': status, 'text': text, 'raw': _raw_strings(r.text) if html else r.text, 'blocked': blocked, 'transient': status in TRANSIENT, 'via': 'http'}


def host(url: str) -> str:
	return urlparse(url).netloc.lower().removeprefix('www.')
