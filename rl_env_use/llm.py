"""Model registry, pricing, structured calls. No OpenRouter or Browser Use key: every route runs on what is already here.

claude     your Claude account through the local Claude Code CLI (`claude -p`)
codex      your ChatGPT account through the local Codex CLI (`codex exec`)
modal      open-weight models served by vLLM on a Modal GPU: started on first use, stopped when the command ends or idles
anthropic  the Anthropic API, only when ANTHROPIC_API_KEY is set (a fallback route for the Claude models)
"""

import asyncio
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, ValidationError

from . import cloud, net

NEEDS = {
	'claude': 'the Claude Code CLI (`claude`) on PATH, signed in',
	'codex': 'the Codex CLI (`codex`) on PATH, signed in',
	'modal': 'a Modal token (`modal token new`)',
	'anthropic': 'ANTHROPIC_API_KEY',
}


def available(route: str) -> bool:
	if route in ('claude', 'codex'):
		return bool(shutil.which(route))
	if route == 'modal':
		return cloud.configured()
	return bool(os.getenv('ANTHROPIC_API_KEY')) if route == 'anthropic' else False


# alias -> candidate (route, model id), first available route wins; the next one is the fallback for an outage
MODELS = {
	'haiku': [('claude', 'claude-haiku-4-5'), ('anthropic', 'claude-haiku-4-5')],
	'sonnet46': [('claude', 'claude-sonnet-4-6'), ('anthropic', 'claude-sonnet-4-6')],
	'sonnet5': [('claude', 'claude-sonnet-5'), ('anthropic', 'claude-sonnet-5')],
	'sonnet55': [('claude', 'claude-sonnet-5-5'), ('anthropic', 'claude-sonnet-5-5')],
	'opus5': [('claude', 'claude-opus-5'), ('anthropic', 'claude-opus-5')],
	'opus55': [('claude', 'claude-opus-5-5'), ('anthropic', 'claude-opus-5-5')],
	'fable': [('claude', 'claude-fable-5-1'), ('anthropic', 'claude-fable-5-1')],
	# the latest of each family, whatever the Claude Code CLI resolves them to
	'claude-sonnet': [('claude', 'sonnet')],
	'claude-opus': [('claude', 'opus')],
	'claude-haiku': [('claude', 'haiku')],
	# OpenAI's frontier through your ChatGPT account (difficulty panel, adjudication)
	'gpt': [('codex', 'gpt-5.5')],
	'gpt-astra': [('codex', 'gpt-6-astra')],
	'gpt6': [('codex', 'gpt-6.1-sol')],
	# open-weight policies to train, served on Modal (any Hugging Face model also works as modal:<org>/<name>)
	'qwen-27b': [('modal', 'Qwen/Qwen3.8-27B-FP8')],
	'qwen-35b': [('modal', 'Qwen/Qwen3.6-35B-A3B-FP8')],
	# envs mined before the switch name Browser Use's gateway models: their missions now run on Claude
	'bu': [('claude', 'claude-sonnet-4-6')],
	'bu-mini': [('claude', 'claude-haiku-4-5')],
}
# what runs before the switch recorded as each alias's model (OpenRouter / Browser Use gateway ids): their slots stay theirs
LEGACY_IDS = {
	'haiku': ['anthropic/claude-haiku-4.5', 'anthropic/claude-haiku-4-5'],
	'sonnet46': ['anthropic/claude-sonnet-4.6', 'anthropic/claude-sonnet-4-6'],
	'sonnet5': ['anthropic/claude-sonnet-5'],
	'opus5': ['anthropic/claude-opus-5'],
	'opus55': ['anthropic/claude-opus-5.5'],
	'sonnet55': ['anthropic/claude-sonnet-5.5'],
	'gpt': ['openai/gpt-5.5'],
	'gpt-astra': ['openai/gpt-6-astra'],
	'gpt6': ['openai/gpt-6.1-sol'],
	'qwen-27b': ['qwen/qwen3.8-27b'],
	'qwen-35b': ['qwen/qwen3.6-35b-a3b'],
	'bu': ['bu-2-0'],
	'bu-mini': ['bu-2-0-mini-preview'],
}
LABELS = {'claude': 'claude-cli', 'codex': 'codex', 'modal': 'modal', 'anthropic': 'anthropic'}  # the route prefix a result row records


def model_ids(alias: str) -> set[str]:
	"""Every model id a result row may carry for this alias, on any route, now or before the switch."""
	return {alias, *LEGACY_IDS.get(alias, [])} | {x for r, m in MODELS.get(alias, []) for x in (m, f'{LABELS[r]}:{m}')}


# vLLM serving per open-weight model: GPU and flags (RLENV_MODAL_GPU / RLENV_VLLM_ARGS override for any model)
QWEN_ARGS = ['--max-model-len', '131072', '--gpu-memory-utilization', '0.92', '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder']
QWEN_ARGS += ['--language-model-only', '--enable-prefix-caching']
SERVING = {'Qwen/Qwen3.8-27B-FP8': ('H100', QWEN_ARGS), 'Qwen/Qwen3.6-35B-A3B-FP8': ('H100', QWEN_ARGS)}
# Frontier closed models: they reason internally at high effort and return only the answer. They decline requests to
# write their reasoning out (a safeguard against copying their outputs), and the grade needs only the answer.
ANSWER_ONLY = {'opus55', 'sonnet55', 'gpt-astra', 'fable'}
# $/MTok (input, output), matched by substring in order. Cache reads 0.1x input, writes 1.25x input. `claude -p` reports
# its API-equivalent cost and Modal models carry their GPU time, so these rows price codex calls, rank models by
# strength (audit) and price old rows.
PRICES = [
	('bu-2-0-mini', None),
	('bu-2-0', (0.6, 3.5)),
	('sonnet-4', (3.0, 15.0)),
	('sonnet', (2.0, 10.0)),
	('haiku-4', (1.0, 5.0)),
	('haiku', (0.1, 0.5)),
	('opus-5.5', (4.0, 20.0)),
	('opus-5-5', (4.0, 20.0)),
	('opus', (5.0, 25.0)),
	('fable', (10.0, 50.0)),
	('gpt-5.5', (5.0, 30.0)),
	('gpt-6-astra', (10.0, 50.0)),
	('gpt-6', (2.0, 10.0)),
	('gemini-3.1-pro', (2.0, 12.0)),
	('gemini-3.8-flash', (0.75, 3.75)),
	('grok-4', (2.0, 6.0)),
	('deepseek-v4-pro', (0.21, 0.42)),
	('qwen3.8-27b', (0.43, 2.55)),
	('qwen3.6-35b', (0.15, 1.0)),
]


class ModelProviderError(RuntimeError):
	def __init__(self, message: str, status_code: int = 500, model: str = ''):
		super().__init__(message)
		self.message, self.status_code, self.model = message, status_code, model


@dataclass
class SystemMessage:
	content: str
	role: str = 'system'


@dataclass
class UserMessage:
	content: str
	role: str = 'user'


def routes(name: str) -> list[tuple[str, str]]:
	"""Every available route of an alias, in preference order; an explicit 'route:model_id' is its only route."""
	if ':' in name and name.split(':', 1)[0] in NEEDS:
		return [tuple(name.split(':', 1))]
	return [(r, m) for r, m in MODELS.get(name, []) if available(r)]


def resolve(name: str) -> tuple[str, str]:
	"""alias ('haiku') or explicit 'route:model_id' ('claude:claude-opus-5-5', 'modal:Qwen/Qwen3.6-27B')."""
	if rs := routes(name):
		return rs[0]
	need = [NEEDS[r] for r, _ in MODELS.get(name, [])] if name in MODELS else [f'a known alias ({", ".join(MODELS)}) or route:model_id with route in {list(NEEDS)}']
	raise RuntimeError(f'no route for model {name!r} (need {" or ".join(dict.fromkeys(need))})')


def first_json(txt: str):
	"""The first complete JSON object in a reply that may wrap it in prose, code fences or a repeated 'Final answer: {...}'."""
	dec = json.JSONDecoder()
	for i, ch in enumerate(txt):
		if ch == '{':
			try:
				return dec.raw_decode(txt, i)[0]
			except ValueError:
				continue
	raise ValueError(f'no JSON object in: {txt[:120]}')


def loads_lenient(txt: str):
	"""JSON as models and gateways actually send it: plain, double-escaped (\\" inside), or Python-literal style."""
	import ast

	for attempt in (lambda t: json.loads(t), lambda t: json.loads(json.loads(f'"{t}"')), ast.literal_eval):
		try:
			return attempt(txt)
		except (ValueError, SyntaxError):
			continue
	raise ValueError(f'unparseable structured field: {txt[:120]}')


def schema_prompt(output_format) -> str:
	return f'\n\nReply with ONLY a JSON object matching this JSON schema (no prose, no code fence):\n{json.dumps(output_format.model_json_schema())}'


def parse_structured(txt: str, output_format):
	txt = txt.strip()
	try:
		obj = loads_lenient(txt[txt.find('{') : txt.rfind('}') + 1] if '{' in txt else txt)
	except ValueError:
		obj = first_json(txt)  # JSON followed by more text, e.g. a repeated final answer
	return output_format.model_validate(obj)


def split_messages(messages) -> tuple[str, str]:
	system = '\n'.join(str(m.content) for m in messages if getattr(m, 'role', '') == 'system')
	return system, '\n\n'.join(str(m.content) for m in messages if getattr(m, 'role', '') != 'system')


# Claude Code without the user's settings, hooks, MCP servers, skills or session files: a plain model call
CLAUDE_BARE = ['--setting-sources', '', '--strict-mcp-config', '--disable-slash-commands', '--no-session-persistence']


class ClaudeCLI:
	"""Plain structured calls through `claude -p` (no tools, our own system prompt), billed to your Claude account. The
	reported cost is the API-equivalent list price: it is what the budget counts."""

	sem = asyncio.Semaphore(int(os.getenv('RLENV_CLAUDE_CONCURRENCY', '8')))

	def __init__(self, model: str):
		self.model = f'claude-cli:{model}'
		self.cli_model = model

	async def ainvoke(self, messages, output_format=None, **kwargs):
		system, user = split_messages(messages)
		if output_format is not None:
			user += schema_prompt(output_format)
		args = ['claude', '-p', '--model', self.cli_model, '--output-format', 'json', '--system-prompt', system or 'You are a precise assistant.', '--tools', '', *CLAUDE_BARE]
		async with self.sem:  # prompt on stdin: a long prompt as an argument outgrows the OS limit
			proc = await asyncio.create_subprocess_exec(
				*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=tempfile.gettempdir()
			)
			try:
				out, err = await asyncio.wait_for(proc.communicate(user.encode()), CALL_TIMEOUT)
			except (TimeoutError, asyncio.CancelledError):
				proc.kill()
				raise
		try:
			w = json.loads(out or b'{}')
		except ValueError:
			w = {'is_error': True, 'result': (out or err).decode()[:300]}
		if w.get('is_error') or 'result' not in w:
			raise ModelProviderError(f'claude -p failed: {(w.get("result") or err.decode())[:200]}', 500, self.model)
		usage = claude_usage(w)
		if output_format is None:
			return SimpleNamespace(completion=w['result'], usage=usage)
		try:
			return SimpleNamespace(completion=parse_structured(w['result'], output_format), usage=usage)
		except Exception:
			LEDGER.add(self.model, usage)  # an unparseable reply is still billed; call() books only the one it returns
			raise


def claude_usage(w: dict) -> dict:
	u = w.get('usage') or {}
	read, write = u.get('cache_read_input_tokens', 0) or 0, u.get('cache_creation_input_tokens', 0) or 0
	return {
		'input_tokens': (u.get('input_tokens', 0) or 0) + read + write,
		'output_tokens': u.get('output_tokens', 0) or 0,
		'cache_read_input_tokens': read,
		'cache_creation_input_tokens': write,
		'reported_cost': w.get('total_cost_usd') or 0,
	}


CODEX_BARE = ['--json', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config', '--ignore-rules']


def codex_events(out: bytes) -> tuple[str, dict, str]:
	"""(last agent message, usage, error) from `codex exec --json` events."""
	text, usage, error = '', {}, ''
	for line in out.decode(errors='replace').splitlines():
		d = json.loads(line) if line.startswith('{') else {}
		if d.get('type') == 'item.completed' and d.get('item', {}).get('type') == 'agent_message':
			text = d['item'].get('text') or ''
		elif d.get('type') == 'turn.completed':
			u = d.get('usage') or {}
			usage = add_usage(
				usage, {'input_tokens': u.get('input_tokens', 0), 'output_tokens': u.get('output_tokens', 0), 'cache_read_input_tokens': u.get('cached_input_tokens', 0)}
			)
		elif d.get('type') in ('error', 'turn.failed'):
			error = str(d.get('message') or d.get('error') or d)[:300]
	return text, usage, error


class CodexCLI:
	"""Plain structured calls through `codex exec` (read-only sandbox, none of the user's config), billed to your ChatGPT
	account. Codex reports tokens, not dollars: PRICES turns them into the API-equivalent cost the budget counts."""

	sem = asyncio.Semaphore(int(os.getenv('RLENV_CODEX_CONCURRENCY', '4')))

	def __init__(self, model: str, effort: str | None = None):
		self.model, self.cli_model, self.effort = f'codex:{model}', model, effort

	async def ainvoke(self, messages, output_format=None, **kwargs):
		system, user = split_messages(messages)
		prompt = (f'{system}\n\n' if system else '') + user + (schema_prompt(output_format) if output_format is not None else '')
		args = ['codex', 'exec', *CODEX_BARE, '-s', 'read-only', '-m', self.cli_model, *(['-c', f'model_reasoning_effort="{self.effort}"'] if self.effort else []), '-']
		async with self.sem:
			proc = await asyncio.create_subprocess_exec(
				*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=tempfile.gettempdir()
			)
			try:
				out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), CALL_TIMEOUT)
			except (TimeoutError, asyncio.CancelledError):
				proc.kill()
				raise
		text, usage, error = codex_events(out)
		if proc.returncode or error or not text:
			raise ModelProviderError(f'codex exec failed: {(error or err.decode() or "no reply")[:200]}', 500, self.model)
		if output_format is None:
			return SimpleNamespace(completion=text, usage=usage)
		try:
			return SimpleNamespace(completion=parse_structured(text, output_format), usage=usage)
		except Exception:
			LEDGER.add(self.model, usage)
			raise


class ModalServer:
	"""One vLLM server per open-weight model, started on first use and shared by every call of the command. Modal stops
	it after RLENV_LLM_IDLE_S without requests (then the next call starts it again) and at the end of the command."""

	servers: dict = {}
	BOOT_TIMEOUT = 45 * 60  # first start of a large model: image build, weight download, CUDA graph capture

	def __init__(self, model: str):
		self.model, self.live, self.base, self.key, self.inflight, self.ready = model, None, None, None, 0, False
		self.loop, self.lock = asyncio.get_running_loop(), asyncio.Lock()

	@classmethod
	async def get(cls, model: str) -> 'ModalServer':
		s = cls.servers.get(model)
		if s is None or s.loop is not asyncio.get_running_loop():  # each command's event loop starts its own (the last one stopped with it)
			s = cls.servers[model] = cls(model)
		await s.ensure()
		return s

	def serving(self) -> tuple[str, list[str]]:
		gpu, args = SERVING.get(self.model, ('H100', ['--max-model-len', '65536', '--enable-prefix-caching']))
		return os.getenv('RLENV_MODAL_GPU', gpu), (os.getenv('RLENV_VLLM_ARGS', '').split() or args)

	async def probe(self) -> bool:
		"""Serving for real: the model is listed and a one-token completion comes back. vLLM answers /health while its
		engine is still compiling kernels, and a request then is reset."""
		import httpx

		h = {'Authorization': f'Bearer {self.key}'}
		try:
			r = await asyncio.to_thread(httpx.get, f'{self.base}/v1/models', headers=h, timeout=15)
			if r.status_code != 200 or self.model not in r.text:
				return False
			body = {'model': self.model, 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 1}
			r = await asyncio.to_thread(httpx.post, f'{self.base}/v1/chat/completions', headers=h, json=body, timeout=120)
			return r.status_code == 200
		except Exception:
			return False

	async def ensure(self) -> None:
		async with self.lock:
			alive = self.live is not None and self.live.sb.object_id in cloud.LIVE and await self.live.sb.poll.aio() is None
			if alive and self.ready:
				return
			if not alive:  # first use, or Modal stopped it when idle: start it (again)
				await cloud.terminate(self.live)
				gpu, args = self.serving()
				self.live, self.base, self.key = await cloud.vllm_sandbox(self.model, gpu, args)
			self.ready, t0 = False, time.monotonic()
			while time.monotonic() - t0 < self.BOOT_TIMEOUT:
				if (rc := await self.live.sb.poll.aio()) is not None:
					logs = ''
					try:
						logs = (await self.live.sb.stderr.read.aio())[-800:]
					except Exception:
						pass
					await cloud.terminate(self.live)
					raise ModelProviderError(f'vLLM for {self.model} exited with {rc}: {logs}', 500, f'modal:{self.model}')
				if await self.probe():
					self.ready = True
					return
				await asyncio.sleep(10)
			await cloud.terminate(self.live)
			raise ModelProviderError(f'vLLM for {self.model} not ready after {self.BOOT_TIMEOUT // 60} min', 504, f'modal:{self.model}')


class OpenAICompat:
	"""Chat completions on an OpenAI-compatible server: a Modal vLLM server, or the Anthropic API's compatible endpoint.
	Structured calls carry the schema in the prompt (not every server enforces response_format) and parse leniently;
	agents use `chat` with tools. A Modal call carries its share of the GPU time as its cost."""

	sem = asyncio.Semaphore(int(os.getenv('RLENV_OPENAI_CONCURRENCY', '16')))

	def __init__(self, route: str, model: str, temperature: float | None = None):
		self.route, self.mid, self.temperature = route, model, temperature
		self.model = f'{route}:{model}'

	async def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 32000) -> tuple[dict, dict, str]:
		"""(assistant message, usage, finish reason)."""
		server = None
		if self.route == 'modal':
			server = await ModalServer.get(self.mid)
			url, headers = f'{server.base}/v1/chat/completions', {'Authorization': f'Bearer {server.key}'}
		else:
			url, headers = 'https://api.anthropic.com/v1/chat/completions', {'Authorization': f'Bearer {os.environ["ANTHROPIC_API_KEY"]}'}
		body = {'model': self.mid, 'messages': messages, 'max_tokens': max_tokens} | ({'temperature': self.temperature} if self.temperature is not None else {})
		body |= {'tools': tools, 'tool_choice': 'auto'} if tools else {}
		async with self.sem:
			t0, share = time.monotonic(), 0.0
			if server:
				server.inflight += 1
			try:
				r = await net.post(url, body, headers, CALL_TIMEOUT)
			except Exception as e:
				if server and not isinstance(e, TimeoutError):  # a dropped connection: the next attempt re-checks the server first
					server.ready = False
				raise ModelProviderError(f'{self.route} connection: {type(e).__name__}: {e}'[:300], 503, self.model) from e
			finally:
				if server:
					share = server.live.rate * (time.monotonic() - t0) / max(1, server.inflight)  # concurrent requests split the GPU's time
					server.inflight -= 1
					server.live.allocated += share
		w = r.json() if r.headers.get('content-type', '').startswith('application/json') else {'error': {'message': r.text[:300]}}
		if r.status_code != 200 or w.get('error') or not w.get('choices'):
			raise ModelProviderError(f'{self.route} {r.status_code}: {str(w.get("error") or w)[:300]}', r.status_code, self.model)
		u = w.get('usage') or {}
		usage = {
			'input_tokens': u.get('prompt_tokens', 0),
			'output_tokens': u.get('completion_tokens', 0),
			'cache_read_input_tokens': (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0,
		} | ({'reported_cost': share} if server else {})
		return w['choices'][0]['message'], usage, w['choices'][0].get('finish_reason') or ''

	async def ainvoke(self, messages, output_format=None, **kwargs):
		msgs = [{'role': getattr(m, 'role', 'user'), 'content': str(m.content)} for m in messages]
		if output_format is not None:
			msgs[-1]['content'] += schema_prompt(output_format)
		msg, usage, finish = await self.chat(msgs)
		txt = (msg.get('content') or '').strip()
		if finish == 'content_filter' or (not txt and msg.get('refusal')):
			raise ModelProviderError('stop_reason refusal: declined to respond', 400, self.model)
		if output_format is None:
			return SimpleNamespace(completion=txt, usage=usage)
		try:
			return SimpleNamespace(completion=parse_structured(txt, output_format), usage=usage)
		except Exception:
			LEDGER.add(self.model, usage)
			raise


def make_llm(name: str, temperature: float | None = None, fallback: bool = False):
	"""A client for plain structured calls. fallback: the alias's next route (the same model elsewhere)."""
	rs = routes(name) or [resolve(name)]
	route, mid = rs[min(int(fallback), len(rs) - 1)]
	if route == 'claude':
		return ClaudeCLI(mid)
	if route == 'codex':
		return CodexCLI(mid, 'high' if name in ANSWER_ONLY else None)
	return OpenAICompat(route, mid, temperature)


def price_of(model: str) -> tuple[float, float] | None:
	m = (model or '').lower()
	return next((p for k, p in PRICES if k in m), None) if m and not m.startswith('modal') else None  # Modal: GPU time, not tokens


def cost(model: str, usage: dict) -> float | None:
	if (usage or {}).get('reported_cost'):
		return usage['reported_cost']
	p = price_of(model)
	if not p or not usage:
		return None
	i, o = p
	read = usage.get('cache_read_input_tokens', 0) or 0
	write = usage.get('cache_creation_input_tokens', 0) or 0
	fresh = (usage.get('input_tokens', 0) or 0) - read - write
	return (fresh * i + read * 0.1 * i + write * 1.25 * i + (usage.get('output_tokens', 0) or 0) * o) / 1e6


def usage_dict(u) -> dict:
	if u is None:
		return {}
	return dict(u)  # every client returns a plain dict, with the billed cost when the route reports one


def add_usage(a: dict, b: dict) -> dict:
	return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in set(a) | set(b)}


@dataclass
class Ledger:
	"""Running spend across every call in a command."""

	by_model: dict = field(default_factory=dict)

	def add(self, model: str, usage: dict):
		if usage and not usage.get('reported_cost'):  # priced per call: a few billed replies must not hide the unbilled ones' tokens in the sum
			usage = usage | {'reported_cost': cost(model, usage) or 0}
		self.by_model[model] = add_usage(self.by_model.get(model, {}), usage)

	@property
	def usd(self) -> float:
		return sum(cost(m, u) or 0 for m, u in list(self.by_model.items()))  # read from the board's thread while calls add to it


LEDGER = Ledger()


CALL_TIMEOUT = 600  # a hung connection or CLI otherwise stalls a whole stage (long structured outputs take minutes)


async def call(name: str, system: str, user: str, schema: type[BaseModel], retries: int = 4, temperature: float | None = 0.0):
	"""One structured LLM call. Returns (parsed, usage, model). Jittered backoff on transient errors; an invalid output
	is fed back to the model on the next attempt (e.g. an enum value outside the allowed set)."""
	feedback = ''
	for attempt in range(retries):
		# a deterministic (temperature 0) failure would repeat identically: later attempts resample
		# the last attempts go to the alias's next route (the same model elsewhere): one route's outage is not the model's
		llm = make_llm(name, temperature if attempt == 0 or temperature is None else 0.7, fallback=attempt >= retries - 2)  # None: the model's own sampling, kept
		try:
			r = await asyncio.wait_for(llm.ainvoke([SystemMessage(content=system), UserMessage(content=user + feedback)], output_format=schema), CALL_TIMEOUT)
			u = usage_dict(r.usage)
			LEDGER.add(llm.model, u)
			return r.completion, u, llm.model
		except Exception as e:
			if attempt == retries - 1:
				raise
			if isinstance(e, ValidationError) or 'validation error' in str(e):
				feedback = f'\n\nYour previous output was invalid - fix this and follow the schema exactly:\n{str(e)[:600]}'
			limited = any(k in str(e).lower() for k in ('rate limit', 'usage limit', 'overloaded', ' 429'))  # an account's rate limit: wait it out
			await asyncio.sleep((60 if limited else min(30, 2**attempt)) * (0.5 + random.random()))


def now() -> str:
	return time.strftime('%Y-%m-%dT%H:%M:%S')


def load_dotenv() -> None:
	"""Keys from ./.env, then from the repo's .env (an editable install of `rl-env-use` works from any folder). Exported vars win."""
	for path in (Path('.env'), Path(__file__).resolve().parents[1] / '.env'):
		for line in path.read_text().splitlines() if path.is_file() else []:
			k, _, v = line.strip().partition('=')
			if k and not k.startswith('#') and v:
				os.environ.setdefault(k, v.strip('"\''))
