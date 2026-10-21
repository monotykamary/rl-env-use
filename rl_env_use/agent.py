"""Browser agents without a framework: a model works a browser-harness-js browser through a small tool set (navigate,
read, snapshot, click, type, ...) plus the caller's own tools (record_fact, ...), and finishes with `done`, whose
arguments are the structured output. One tool set, three drivers:

claude / codex     the CLI runs its own agent loop; the tools reach it as an MCP server (bridge.py) that relays each
                   call to this process, where it runs against the live browser and the kb
modal / anthropic  our own loop over OpenAI-compatible chat completions with tool calls

Every call goes through `Tools.call`, so the step budget, the trace and the visited pages are the same for every driver.
"""

import asyncio
import inspect
import json
import re
import secrets
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from .browse import EXPAND_JS, Browser, BrowserError
from .llm import CLAUDE_BARE, CODEX_BARE, LABELS, LEDGER, OpenAICompat, add_usage, claude_usage, codex_events, resolve

RESULT_CHARS = 20000  # one tool result the model sees
STEP_GRACE = 5  # calls past the step budget (each answered "call done now") before a CLI agent is stopped

AGENT_SYS = """You operate a real web browser through tools; what the tools return is all you know about a page.
- navigate opens a URL. read_page returns the page text (find="words" greps it; start=N reads further). list_links lists links.
- snapshot shows the interactive elements with refs; click / type_text / select_option take a ref from the LATEST snapshot (refs expire after any action: take a new snapshot).
- expand_page opens collapsed sections, cookie walls and "show more" buttons; scroll loads lazy content.
- Never invent page content or URLs you have not seen. When the task is finished, call `done` exactly once."""


class ToolError(Exception):
	"""Shown to the model as an error result: it reads the message and tries again."""


@dataclass
class Tool:
	name: str
	description: str
	fn: Any  # async (**params) -> str
	params: type[BaseModel]

	def schema(self) -> dict:
		s = self.params.model_json_schema()
		s.pop('title', None)
		for p in s.get('properties', {}).values():
			p.pop('title', None)
		return s


def tool(description: str, name: str | None = None):
	"""An async function as a tool: its keyword parameters (annotations and defaults) are the tool's arguments."""

	def wrap(fn) -> Tool:
		fields = {p.name: (p.annotation, ... if p.default is p.empty else p.default) for p in inspect.signature(fn).parameters.values()}
		return Tool(name or fn.__name__, description, fn, create_model(f'{fn.__name__}_args', **fields))

	return wrap


@dataclass
class Run:
	steps: int
	done: bool
	output: BaseModel | None
	urls: list[str]
	trace: list[dict]
	errors: list[str]
	usage: dict
	model: str
	stop_reason: str


@dataclass
class Tools:
	"""The tools of one run, with its step budget, trace and visited pages."""

	tools: list[Tool]
	browser: Browser
	max_steps: int
	output: type[BaseModel]
	done_description: str = 'Finish: give the final result. Call exactly once, as the last action.'
	steps: int = 0
	result: BaseModel | None = None
	trace: list[dict] = field(default_factory=list)
	urls: list[str] = field(default_factory=list)
	errors: list[str] = field(default_factory=list)

	def __post_init__(self):
		self.by_name = {t.name: t for t in self.tools} | {'done': Tool('done', self.done_description, None, self.output)}

	@property
	def over(self) -> bool:
		return self.result is not None or self.steps > self.max_steps + STEP_GRACE

	def specs(self) -> list[dict]:
		return [{'name': t.name, 'description': t.description, 'inputSchema': t.schema()} for t in self.by_name.values()]

	def note(self, text: str, thinking: str | None = None) -> None:
		"""The model's own words between calls (its plan, its reasoning) go into the trace as an assistant turn."""
		if (text or '').strip() or thinking:
			self.trace.append({'role': 'assistant', 'content': (text or '').strip(), 'thinking': thinking or None})

	async def call(self, name: str, args: dict) -> tuple[str, bool]:
		if self.result is not None:
			return 'Already done: stop now.', False
		if self.steps >= self.max_steps and name != 'done':
			self.steps += 1
			return f'Step budget ({self.max_steps}) spent: call done now with what you have.', True
		self.steps += 1
		t, err = self.by_name.get(name), True
		try:
			if t is None:
				raise ToolError(f'Unknown tool {name!r}: use one of {sorted(self.by_name)}')
			params = t.params.model_validate(args or {})
			if name == 'done':
				self.result, text, err = params, 'Done. Stop now.', False
			else:
				text, err = str(await t.fn(**dict(params))), False
		except ValidationError as e:
			text = f'Invalid arguments for {name}: {e}'[:1500]
		except (ToolError, BrowserError) as e:
			text = str(e)
		except Exception as e:  # a tool crash is shown to the model, and kept as an error of the run
			text = f'{type(e).__name__}: {e}'[:1000]
			self.errors.append(text)
		try:
			url = await self.browser.url() if self.browser.url_ else ''
		except Exception:
			url = ''
		if url.startswith('http'):
			self.urls.append(url)
		self.trace += [
			{'role': 'assistant', 'content': f'[step {self.steps} @ {url or "about:blank"}]'},
			{'role': 'tool_call', 'name': name, 'content': json.dumps(args, indent=1, default=str)},
			{'role': 'tool_result', 'content': ('ERROR: ' if err else '') + text[:RESULT_CHARS]},
		]
		return text[:RESULT_CHARS], err


def browser_tools(b: Browser) -> list[Tool]:
	@tool('Open a URL in the browser (http or https). Returns the page title and the start of its text.')
	async def navigate(url: str) -> str:
		if not re.match(r'https?://', url):
			raise ToolError('Give a full http(s) URL.')
		w = await b.goto(url)
		text = await b.text()
		return f'Opened {w["url"]} - {w["title"]} ({len(text)} chars of text)\n\n{text[:3000]}'

	@tool(
		'Read the text of the current page, 12000 characters from offset `start`. `find`: instead list where these words occur (case-insensitive, at or after `start`) with the offset of each hit, so read_page(start=offset) reads from there.'
	)
	async def read_page(find: str = '', start: int = 0) -> str:
		text = await b.text()
		if find.strip():
			words, at, hits = find.lower().split(), 0, []
			for ln in text.splitlines(keepends=True):
				if at + len(ln) > start and all(w in ln.lower() for w in words):
					hits.append(f'@{at}: {" ".join(ln.split())[:400]}')
				at += len(ln)
			if not hits:
				return f'No line on this page contains {find!r} after offset {start} ({len(text)} chars). Try other words, expand_page, or scroll.'
			return '\n'.join(hits[:40]) + f'\n[{len(hits)} matching lines{", first 40 shown" if len(hits) > 40 else ""}; read_page(start=<offset>) reads the text from a hit]'
		end = start + 12000
		return text[start:end] + (f'\n[chars {start}-{min(end, len(text))} of {len(text)}; read_page(start={end}) reads on]' if end < len(text) else '')

	@tool('The interactive elements of the current page (accessibility tree) with refs for click / type_text / select_option.')
	async def snapshot() -> str:
		tree = await b.snapshot(True)
		return tree[:15000] + ('\n[snapshot truncated: use read_page / list_links for the rest]' if len(tree) > 15000 else '')

	@tool('Click the element with this ref from the latest snapshot.')
	async def click(ref: str) -> str:
		r = await b.act(ref, 'click')
		return f'{r["status"]} ({r["reason"]}); now at {r["url"]} - {r["title"]}'

	@tool('Replace the text of the input with this ref (from the latest snapshot) by `text`; submit=true then presses Enter.')
	async def type_text(ref: str, text: str, submit: bool = False) -> str:
		r = await b.act(ref, 'fill', text)
		if submit and r['status'] == 'executed':
			await b.js(
				"await cdp(rl.sid, 'Input.dispatchKeyEvent', {type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\\r'}); await cdp(rl.sid, 'Input.dispatchKeyEvent', {type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13}); await rl.settle(10000); return true;"
			)
			w = await b.call('where')
			return f'typed and submitted; now at {w["url"]} - {w["title"]}'
		return f'{r["status"]} ({r["reason"]}, verified={r["verified"]})'

	@tool('Choose an option (its visible label, or a 0-based index) in the select with this ref from the latest snapshot.')
	async def select_option(ref: str, option: str) -> str:
		r = await b.act(ref, 'select', int(option) if option.isdigit() else option)
		return f'{r["status"]} ({r["reason"]}, verified={r["verified"]}); now at {r["url"]}'

	@tool('Go back to the previous page in this tab.')
	async def go_back() -> str:
		w = await b.call('back')
		return f'Back at {w["url"]} - {w["title"]}'

	@tool('Scroll the page down (or up) by one screen, which also loads lazy content.')
	async def scroll(down: bool = True) -> str:
		y = await b.eval(f'(window.scrollBy(0, {"" if down else "-"}0.9 * innerHeight), Math.round(scrollY))')
		return f'scrolled to y={y} of {await b.eval("document.documentElement.scrollHeight")}'

	@tool('Open collapsed sections, accept cookie walls and press "show more" / "load more" buttons until the page stops growing.')
	async def expand_page() -> str:
		clicks, size = 0, len(await b.text())
		for _ in range(6):
			n = await b.eval(EXPAND_JS) or 0
			clicks += n
			await asyncio.sleep(1.5)
			new = len(await b.text())
			if not n and new <= size:
				break
			size = max(size, new)
		return f'{clicks} controls expanded; the page text is now {size} chars'

	@tool('List the links on the current page as "text -> url". `find`: only links whose text or url contains these words.')
	async def list_links(find: str = '') -> str:
		links = (
			await b.eval(
				'[...document.querySelectorAll("a[href]")].map(a => [(a.innerText || a.getAttribute("aria-label") || "").trim().replace(/\\s+/g, " ").slice(0, 120), a.href])'
			)
			or []
		)
		words = find.lower().split()
		rows = list(dict.fromkeys(f'{t} -> {u}' for t, u in links if u.startswith('http') and all(w in f'{t} {u}'.lower() for w in words)))
		return '\n'.join(rows[:150]) + (f'\n[{len(rows)} links, first 150 shown: narrow with find]' if len(rows) > 150 else '') if rows else 'No matching links.'

	return [navigate, read_page, snapshot, click, type_text, select_option, go_back, scroll, expand_page, list_links]


class Relay:
	"""The loopback end of bridge.py: lists the run's tools and executes their calls in this process."""

	def __init__(self, tools: Tools):
		self.tools, self.token, self.server, self.port = tools, secrets.token_hex(16), None, 0

	async def __aenter__(self):
		self.server = await asyncio.start_server(self.handle, '127.0.0.1', 0, limit=2**24)
		self.port = self.server.sockets[0].getsockname()[1]
		return self

	async def __aexit__(self, *exc):
		self.server.close()

	async def handle(self, reader, writer):
		try:
			req = json.loads(await reader.readline() or b'{}')
			if req.get('token') != self.token:
				return
			if req.get('op') == 'list':
				resp = {'tools': self.tools.specs()}
			else:
				text, err = await self.tools.call(req.get('name'), req.get('args') or {})
				resp = {'text': text, 'is_error': err}
			writer.write((json.dumps(resp) + '\n').encode())
			await writer.drain()
		finally:
			writer.close()

	def command(self) -> tuple[str, list[str]]:
		return sys.executable, [str(Path(__file__).with_name('bridge.py')), str(self.port), self.token]


async def cli_agent(argv: list[str], prompt: str, tools: Tools, timeout: float) -> bytes:
	"""Run a CLI agent to the end, or stop it once the run is over (done, or the step budget spent)."""
	proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=tempfile.gettempdir())
	comm = asyncio.ensure_future(proc.communicate(prompt.encode()))
	try:
		loop = asyncio.get_running_loop()
		end, over_at = loop.time() + timeout, None
		while not comm.done():
			await asyncio.wait([comm], timeout=1)
			if tools.over and over_at is None:
				over_at = loop.time()
			if loop.time() > end or (over_at is not None and loop.time() - over_at > 30):  # done but still talking: enough
				proc.kill()
		out, err = await comm
		return out + b'\n' + err
	finally:
		if proc.returncode is None:
			proc.kill()
		if not comm.done():
			comm.cancel()


async def claude_loop(mid: str, system: str, task: str, tools: Tools, timeout: float) -> tuple[dict, str]:
	async with Relay(tools) as relay:
		cmd, args = relay.command()
		mcp = json.dumps({'mcpServers': {'rlenv': {'type': 'stdio', 'command': cmd, 'args': args}}})
		argv = ['claude', '-p', '--model', mid, '--output-format', 'stream-json', '--verbose', '--system-prompt', system, '--tools', '', *CLAUDE_BARE]
		argv += ['--mcp-config', mcp, '--allowedTools', 'mcp__rlenv', '--max-turns', str(tools.max_steps + STEP_GRACE + 2)]
		out = await cli_agent(argv, task, tools, timeout)
	usage, stop, failed = {}, 'done', ''
	for line in out.decode(errors='replace').splitlines():
		d = json.loads(line) if line.startswith('{') else {}
		content = d.get('message', {}).get('content') if d.get('type') == 'assistant' else None
		for blk in content if isinstance(content, list) else []:
			if blk.get('type') == 'text':
				tools.note(blk.get('text', ''))
		if d.get('type') == 'result':
			usage = claude_usage(d)
			stop = 'max_steps' if d.get('subtype') == 'error_max_turns' else stop
			failed = (d.get('result') or d.get('subtype') or 'error') if d.get('is_error') and d.get('subtype') != 'error_max_turns' else ''
	if not usage and not tools.result:
		failed = out.decode(errors='replace')[-300:] or 'no result'
	if failed and tools.steps == 0:  # quota, auth or rate limit before the first action: infra, retried on resume
		raise RuntimeError(f'claude -p agent failed: {failed[:300]}')
	return usage, stop


async def codex_loop(mid: str, system: str, task: str, tools: Tools, timeout: float) -> tuple[dict, str]:
	async with Relay(tools) as relay:
		cmd, args = relay.command()
		argv = ['codex', 'exec', *CODEX_BARE, '-s', 'read-only', '-m', mid, '-c', 'approval_policy="never"']
		argv += ['-c', f'mcp_servers.rlenv.command={json.dumps(cmd)}', '-c', f'mcp_servers.rlenv.args={json.dumps(args)}', '-c', 'mcp_servers.rlenv.tool_timeout_sec=300']
		argv += ['-c', 'mcp_servers.rlenv.default_tools_approval_mode="approve"', '-']  # exec mode never asks: unapproved MCP calls fail
		out = await cli_agent(argv, f'{system}\n\nUse only the rlenv tools (no shell).\n\n{task}', tools, timeout)
	text, usage, error = codex_events(out)
	tools.note(text)
	if error and tools.steps == 0:
		raise RuntimeError(f'codex exec agent failed: {error}')
	return usage, 'done' if tools.result else 'max_steps' if tools.steps >= tools.max_steps else 'stopped'


def compact(msgs: list[dict], keep: int = 4, chars: int = 600) -> None:
	"""Older tool results shrink to their start: the model keeps what it learned in its own notes, not every page."""
	idx = [i for i, m in enumerate(msgs) if m['role'] == 'tool']
	for i in idx[:-keep]:
		if len(msgs[i]['content']) > chars:
			msgs[i]['content'] = msgs[i]['content'][:chars] + ' ...[older result trimmed]'


async def openai_loop(route: str, mid: str, system: str, task: str, tools: Tools) -> tuple[dict, str]:
	llm = OpenAICompat(route, mid)
	specs = [{'type': 'function', 'function': {'name': s['name'], 'description': s['description'], 'parameters': s['inputSchema']}} for s in tools.specs()]
	msgs, usage, nudges = [{'role': 'system', 'content': system}, {'role': 'user', 'content': task}], {}, 0
	while not tools.over:
		compact(msgs)
		for attempt in range(3):
			try:
				msg, u, _ = await llm.chat(msgs, specs, max_tokens=8000)
				break
			except Exception:
				if attempt == 2:
					raise
				await asyncio.sleep(5 * (attempt + 1))
		usage = add_usage(usage, u)
		calls = msg.get('tool_calls') or []
		tools.note(msg.get('content') or '', msg.get('reasoning_content') or msg.get('reasoning'))
		msgs.append({'role': 'assistant', 'content': msg.get('content') or ''} | ({'tool_calls': calls} if calls else {}))
		if not calls:
			nudges += 1
			if nudges > 2:
				break
			msgs.append({'role': 'user', 'content': 'Continue with the tools. When the task is finished, call `done`.'})
			continue
		for c in calls:
			try:
				args = json.loads(c['function'].get('arguments') or '{}')
			except ValueError:
				args = {'_unparseable': c['function'].get('arguments')}
			text, err = await tools.call(c['function']['name'], args)
			msgs.append({'role': 'tool', 'tool_call_id': c['id'], 'content': ('ERROR: ' if err else '') + text})
	return usage, 'done' if tools.result else 'max_steps' if tools.steps >= tools.max_steps else 'stopped'


async def run_agent(
	model: str,
	task: str,
	browser: Browser,
	output: type[BaseModel],
	tools: list[Tool] = (),
	max_steps: int = 30,
	system: str | None = None,
	start_url: str | None = None,
	done_description: str | None = None,
	timeout: float = 1500,
) -> Run:
	"""One agent run on a started browser. The model id in the result names the route (claude-cli:, codex:, modal:, ...)."""
	route, mid = resolve(model)
	ts = Tools([*browser_tools(browser), *tools], browser, max_steps, output, **({'done_description': done_description} if done_description else {}))
	ts.trace.append({'role': 'system', 'content': task})
	if start_url:  # the task's start page is open before the first step, as a person would find it
		await browser.goto(start_url)
	system = AGENT_SYS + (f'\n\n{system}' if system else '')
	label = f'{LABELS[route]}:{mid}'
	usage, stop = {}, 'stopped'
	try:
		if route == 'claude':
			usage, stop = await claude_loop(mid, system, task, ts, timeout)
		elif route == 'codex':
			usage, stop = await codex_loop(mid, system, task, ts, timeout)
		else:
			usage, stop = await openai_loop(route, mid, system, task, ts)
	finally:
		LEDGER.add(label, usage)  # a crashed or cancelled run was still billed for what it used
	return Run(ts.steps, ts.result is not None, ts.result, list(dict.fromkeys(ts.urls)), ts.trace, ts.errors, usage, label, 'done' if ts.result else stop)
