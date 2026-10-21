import asyncio
import json

import pytest
from pydantic import BaseModel

import rl_env_use.llm as L
from rl_env_use import net
from rl_env_use.audit import strength
from rl_env_use.llm import ModelProviderError, SystemMessage, UserMessage
from rl_env_use.runner import REFUSAL


class Ans(BaseModel):
	answer: str
	confidence: float


class Resp:
	def __init__(self, status=200, body=None, ctype='application/json', text=''):
		self.status_code, self._body, self.headers, self.text = status, body, {'content-type': ctype}, text

	def json(self):
		return self._body


def ok(content, finish='stop', usage=True, refusal=None):
	u = {'prompt_tokens': 1000, 'completion_tokens': 200, 'prompt_tokens_details': {'cached_tokens': 100}}
	return Resp(body={'choices': [{'message': {'content': content, 'refusal': refusal}, 'finish_reason': finish}], **({'usage': u} if usage else {})})


@pytest.fixture
def fake(monkeypatch):
	"""Patch the sync httpx.Client the worker threads use; returns a holder with .queue (responses) and .posts (requests)."""
	import httpx

	h = type('H', (), {'queue': [], 'posts': []})()

	class Client:
		is_closed = False

		def __init__(self, timeout=None, **kw):
			self.timeout = timeout

		def post(self, url, headers=None, json=None, content=None, timeout=None):
			h.posts.append({'url': url, 'headers': headers, 'json': json, 'timeout': timeout})
			return h.queue.pop(0)

	monkeypatch.setattr(httpx, 'Client', Client)
	monkeypatch.setattr(net, '_local', __import__('threading').local())
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	return h


def api(model='claude-opus-5-5', temperature=None):
	return L.OpenAICompat('anthropic', model, temperature)


def inv(llm, msgs=None, fmt=Ans):
	return asyncio.run(llm.ainvoke(msgs or [SystemMessage(content='sys'), UserMessage(content='q?')], output_format=fmt))


def test_parses_json_inside_prose_and_fence(fake):
	fake.queue.append(ok('Sure, here it is:\n```json\n{"answer": "137", "confidence": 0.9}\n```\nHope that helps!'))
	r = inv(api())
	assert r.completion == Ans(answer='137', confidence=0.9)
	p = fake.posts[0]
	assert p['url'] == 'https://api.anthropic.com/v1/chat/completions' and p['headers']['Authorization'] == 'Bearer sk-ant-test'
	assert p['json']['model'] == 'claude-opus-5-5' and p['timeout'] == L.CALL_TIMEOUT


def test_parses_python_literal_style(fake):
	fake.queue.append(ok("{'answer': 'x', 'confidence': 0.5}"))
	assert inv(api()).completion.answer == 'x'


def test_plain_text_without_schema(fake):
	fake.queue.append(ok('  just text  '))
	r = inv(api(), fmt=None)
	assert r.completion == 'just text'
	assert 'JSON schema' not in fake.posts[0]['json']['messages'][-1]['content']


def test_invalid_json_shape_raises_validation(fake):
	fake.queue.append(ok('{"answer": "x"}'))
	with pytest.raises(Exception, match='validation error'):
		inv(api())


def test_schema_appended_to_last_message_only(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}'))
	inv(api(), [SystemMessage(content='sys'), UserMessage(content='first'), UserMessage(content='last')])
	msgs = fake.posts[0]['json']['messages']
	assert [m['role'] for m in msgs] == ['system', 'user', 'user']
	assert msgs[0]['content'] == 'sys' and msgs[1]['content'] == 'first'
	assert msgs[2]['content'].startswith('last\n\nReply with ONLY a JSON object')
	assert json.loads(msgs[2]['content'].split('\n')[-1]) == Ans.model_json_schema()


def test_temperature_omitted_when_none(fake):
	fake.queue += [ok('{"answer": "a", "confidence": 1}')] * 3
	inv(api())
	inv(api(temperature=0.0))
	inv(api(temperature=0.7))
	assert 'temperature' not in fake.posts[0]['json']
	assert fake.posts[1]['json']['temperature'] == 0.0 and fake.posts[2]['json']['temperature'] == 0.7


def test_api_usage_is_priced_from_tokens(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}'))
	u = inv(api()).usage
	assert u == {'input_tokens': 1000, 'output_tokens': 200, 'cache_read_input_tokens': 100}
	assert L.usage_dict(u) == u
	assert L.cost('anthropic:claude-opus-5-5', u) == pytest.approx((900 * 4.0 + 100 * 0.4 + 200 * 20.0) / 1e6)


def test_missing_usage_block_is_zeroed(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', usage=False))
	assert inv(api()).usage == {'input_tokens': 0, 'output_tokens': 0, 'cache_read_input_tokens': 0}


@pytest.mark.parametrize(
	'resp,code',
	[
		(Resp(429, {'error': {'message': 'rate limited'}}), 429),
		(Resp(502, None, ctype='text/html', text='<html>bad gateway</html>'), 502),
		(Resp(200, {'error': {'message': 'provider down', 'code': 500}}), 200),
		(Resp(200, {'choices': []}), 200),
	],
)
def test_http_errors_raise_provider_error(fake, resp, code):
	fake.queue.append(resp)
	with pytest.raises(ModelProviderError) as e:
		inv(api())
	assert e.value.status_code == code and f'anthropic {code}' in str(e.value)
	assert not REFUSAL.search(str(e.value))


def test_html_error_body_in_message(fake):
	fake.queue.append(Resp(503, None, ctype='text/plain', text='upstream overloaded'))
	with pytest.raises(ModelProviderError, match='upstream overloaded'):
		inv(api())


@pytest.mark.parametrize('content,finish,refusal', [('partial', 'content_filter', None), ('', 'stop', 'I cannot help'), (None, 'stop', 'no')])
def test_refusals_classified_by_runner(fake, content, finish, refusal):
	fake.queue.append(ok(content, finish=finish, refusal=refusal))
	with pytest.raises(ModelProviderError) as e:
		inv(api())
	assert REFUSAL.search(str(e.value))


def test_refusal_field_with_content_is_not_refusal(fake):
	fake.queue.append(ok('{"answer": "a", "confidence": 1}', refusal='x'))
	assert inv(api()).completion.answer == 'a'


class FakeServer:
	def __init__(self):
		self.base, self.key, self.inflight = 'https://tunnel.modal.host', 'k3y', 0
		self.live = type('Live', (), {'rate': 0.001, 'allocated': 0.0})()


def test_modal_route_posts_to_its_server_and_carries_gpu_time(fake, monkeypatch):
	srv = FakeServer()

	async def get(model):
		assert model == 'Qwen/Qwen3.6-35B-A3B-FP8'
		return srv

	monkeypatch.setattr(L.ModalServer, 'get', staticmethod(get))
	fake.queue.append(ok('{"answer": "a", "confidence": 1}'))
	r = inv(L.OpenAICompat('modal', 'Qwen/Qwen3.6-35B-A3B-FP8'))
	p = fake.posts[0]
	assert p['url'] == 'https://tunnel.modal.host/v1/chat/completions' and p['headers']['Authorization'] == 'Bearer k3y'
	assert r.usage['reported_cost'] == pytest.approx(srv.live.allocated) and srv.inflight == 0
	assert L.price_of('modal:Qwen/Qwen3.6-35B-A3B-FP8') is None  # GPU time, never tokens at another host's price


def test_resolve_routes_by_what_is_installed(monkeypatch):
	monkeypatch.setattr(L.shutil, 'which', lambda c: None)
	monkeypatch.setattr(L.cloud, 'configured', lambda: False)
	monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
	with pytest.raises(RuntimeError, match=r"no route for model 'gpt6'.*Codex CLI"):
		L.resolve('gpt6')
	with pytest.raises(RuntimeError, match=r"no route for model 'qwen-35b'.*Modal token"):
		L.resolve('qwen-35b')
	monkeypatch.setattr(L.shutil, 'which', lambda c: f'/bin/{c}')
	monkeypatch.setattr(L.cloud, 'configured', lambda: True)
	assert L.resolve('opus55') == ('claude', 'claude-opus-5-5')
	assert L.resolve('gpt-astra') == ('codex', 'gpt-6-astra')
	assert L.resolve('qwen-35b') == ('modal', 'Qwen/Qwen3.6-35B-A3B-FP8')
	assert L.resolve('bu') == ('claude', 'claude-sonnet-4-6')  # envs mined on Browser Use name its gateway model


def test_resolve_explicit_routes_and_unknown(monkeypatch):
	assert L.resolve('modal:meta-llama/Llama-5-70B') == ('modal', 'meta-llama/Llama-5-70B')
	assert L.resolve('claude:claude-fable-5-1') == ('claude', 'claude-fable-5-1')
	with pytest.raises(RuntimeError, match='no route'):
		L.resolve('no-such-model')
	with pytest.raises(RuntimeError, match='no route'):
		L.resolve('openrouter:openai/gpt-6.1-sol')  # no OpenRouter route any more


def test_make_llm_picks_the_client(monkeypatch):
	monkeypatch.setattr(L.shutil, 'which', lambda c: f'/bin/{c}')
	assert isinstance(L.make_llm('sonnet46'), L.ClaudeCLI)
	codex = L.make_llm('gpt-astra')
	assert isinstance(codex, L.CodexCLI) and codex.effort == 'high' and L.make_llm('gpt').effort is None
	m = L.make_llm('modal:a/b', temperature=0.3)
	assert isinstance(m, L.OpenAICompat) and m.route == 'modal' and m.temperature == 0.3


def test_call_end_to_end_with_ledger(fake, monkeypatch):
	monkeypatch.setattr(L, 'routes', lambda name: [('anthropic', 'claude-opus-5-5')])
	fake.queue += [ok('{"answer": "a", "confidence": 1}'), ok('```\n{"answer": "b", "confidence": 0.2}\n```')]
	p1, u1, m1 = asyncio.run(L.call('opus55', 'sys', 'u1', Ans))
	p2, u2, m2 = asyncio.run(L.call('opus55', 'sys', 'u2', Ans))
	assert (p1.answer, p2.answer) == ('a', 'b') and m1 == m2 == 'anthropic:claude-opus-5-5'
	assert fake.posts[0]['json']['temperature'] == 0.0 and fake.posts[0]['json']['messages'][1]['content'].startswith('u1')
	assert L.LEDGER.by_model['anthropic:claude-opus-5-5']['input_tokens'] == 2000
	assert L.LEDGER.usd == pytest.approx(2 * (900 * 4.0 + 100 * 0.4 + 200 * 20.0) / 1e6)


def test_call_retries_with_feedback_and_resampling(fake, monkeypatch):
	async def nosleep(s):
		pass

	monkeypatch.setattr(L.asyncio, 'sleep', nosleep)
	monkeypatch.setattr(L, 'routes', lambda name: [('anthropic', 'claude-haiku-4-5')])
	fake.queue += [Resp(500, {'error': {'message': 'boom'}}), ok('{"answer": "x"}'), ok('{"answer": "c", "confidence": 0.5}')]
	p, u, m = asyncio.run(L.call('haiku', 's', 'q', Ans))
	assert p.answer == 'c'
	assert [x['json'].get('temperature') for x in fake.posts] == [0.0, 0.7, 0.7]
	assert 'previous output was invalid' not in fake.posts[1]['json']['messages'][1]['content']
	assert 'previous output was invalid' in fake.posts[2]['json']['messages'][1]['content']
	one = L.cost('anthropic:claude-haiku-4-5', {'input_tokens': 1000, 'output_tokens': 200, 'cache_read_input_tokens': 100})
	assert L.LEDGER.usd == pytest.approx(2 * one)  # the 500 is not billed; the unparseable 200 was, and is booked


def test_call_refusal_propagates_after_retries(fake, monkeypatch):
	async def nosleep(s):
		pass

	monkeypatch.setattr(L.asyncio, 'sleep', nosleep)
	monkeypatch.setattr(L, 'routes', lambda name: [('anthropic', 'claude-haiku-4-5')])
	fake.queue += [ok('', finish='content_filter')] * 2
	with pytest.raises(ModelProviderError) as e:
		asyncio.run(L.call('haiku', 's', 'q', Ans, retries=2))
	assert REFUSAL.search(str(e.value)) and not fake.queue


def test_ledger_mixes_reported_and_priced():
	led = L.Ledger()
	led.add('claude-cli:claude-opus-5-5', {'input_tokens': 10, 'output_tokens': 10, 'reported_cost': 0.5})
	led.add('claude-cli:claude-opus-5-5', {'input_tokens': 10, 'output_tokens': 10, 'reported_cost': 0.25})
	led.add('codex:gpt-6-astra', {'input_tokens': 1_000_000, 'output_tokens': 0})
	assert led.usd == pytest.approx(0.75 + 10.0)
	led.add('modal browser', {'reported_cost': 0.01})
	led.add('modal llm Qwen/Qwen3.6-35B-A3B-FP8', {'reported_cost': 0})  # an unbilled entry stays at zero: Modal time is never priced by tokens
	assert led.usd == pytest.approx(0.75 + 10.0 + 0.01)


def test_strength_orders_models():
	ids = {a: L.MODELS[a][0][1] for a in ('haiku', 'gpt6', 'opus55', 'gpt-astra')}
	s = {a: strength(m) for a, m in ids.items()}
	assert s['haiku'] < s['gpt6'] < s['opus55'] < s['gpt-astra']
	assert strength('unknown/model') == 0.0 and strength(None) == 0.0
	assert strength('google/gemini-3.1-pro-preview') > 0  # rows from earlier runs still rank
	assert all(L.price_of(m) for m in ids.values())


def test_last_attempts_fall_back_to_the_same_model_on_another_route(monkeypatch):
	class A(BaseModel):
		answer: str

	monkeypatch.setattr(L.shutil, 'which', lambda c: f'/bin/{c}')
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'k')
	assert L.routes('sonnet46') == [('claude', 'claude-sonnet-4-6'), ('anthropic', 'claude-sonnet-4-6')]
	seen = []

	class Fake:
		def __init__(self, route, mid):
			self.model, self.route = mid, route

		async def ainvoke(self, messages, output_format=None, **kw):
			seen.append(self.route)
			if self.route == 'claude':
				raise RuntimeError('claude -p failed: overloaded')
			return L.SimpleNamespace(completion=A(answer='ok'), usage={'input_tokens': 1, 'output_tokens': 1, 'reported_cost': 0.01})

	monkeypatch.setattr(L, 'make_llm', lambda name, t=None, fallback=False: Fake(*L.routes(name)[int(fallback)]))
	nap = asyncio.sleep
	monkeypatch.setattr(L.asyncio, 'sleep', lambda *_: nap(0))
	r, u, mid = asyncio.run(L.call('sonnet46', 's', 'u', A))
	assert r.answer == 'ok' and mid == 'claude-sonnet-4-6' and seen == ['claude', 'claude', 'anthropic']


class Proc:
	def __init__(self, out: bytes, rc: int = 0, err: bytes = b''):
		self.out, self.err, self.returncode, self.stdin_seen = out, err, rc, None

	async def communicate(self, data):
		self.stdin_seen = data
		return self.out, self.err

	def kill(self):
		pass


@pytest.fixture
def spawn(monkeypatch):
	h = type('H', (), {'procs': [], 'argv': []})()

	async def exec_(*argv, **kw):
		h.argv.append(argv)
		return h.procs.pop(0)

	monkeypatch.setattr(L.asyncio, 'create_subprocess_exec', exec_)
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	return h


def test_claude_cli_runs_bare_and_books_its_reported_cost(spawn):
	w = {
		'result': 'Here: {"answer": "a", "confidence": 1}',
		'total_cost_usd': 0.042,
		'usage': {'input_tokens': 5, 'cache_read_input_tokens': 90, 'cache_creation_input_tokens': 5, 'output_tokens': 7},
	}
	spawn.procs.append(Proc(json.dumps(w).encode()))
	r = inv(L.ClaudeCLI('claude-opus-5-5'))
	argv = spawn.argv[0]
	assert r.completion.answer == 'a' and r.usage['reported_cost'] == 0.042 and r.usage['input_tokens'] == 100
	assert argv[:4] == ('claude', '-p', '--model', 'claude-opus-5-5') and argv[argv.index('--tools') + 1] == '' and argv[argv.index('--setting-sources') + 1] == ''
	assert '--strict-mcp-config' in argv and argv[argv.index('--system-prompt') + 1] == 'sys'


def test_claude_cli_error_raises(spawn):
	spawn.procs.append(Proc(json.dumps({'is_error': True, 'result': 'usage limit reached'}).encode()))
	with pytest.raises(ModelProviderError, match='usage limit'):
		inv(L.ClaudeCLI('sonnet'))


def test_codex_cli_reads_events_ignores_user_config_and_prices_tokens(spawn):
	events = [
		{'type': 'thread.started'},
		{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'thinking aloud'}},
		{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"answer": "b", "confidence": 0.5}'}},
		{'type': 'turn.completed', 'usage': {'input_tokens': 2000, 'cached_input_tokens': 1000, 'output_tokens': 10}},
	]
	spawn.procs.append(Proc('\n'.join(map(json.dumps, events)).encode()))
	r = inv(L.CodexCLI('gpt-6-astra', 'high'))
	argv = spawn.argv[0]
	assert r.completion.answer == 'b' and r.usage == {'input_tokens': 2000, 'output_tokens': 10, 'cache_read_input_tokens': 1000}
	assert '--ignore-user-config' in argv and '--ephemeral' in argv and argv[argv.index('-s') + 1] == 'read-only' and 'model_reasoning_effort="high"' in argv
	assert spawn.procs == [] and L.cost('codex:gpt-6-astra', r.usage) == pytest.approx((1000 * 10 + 1000 * 1.0 + 10 * 50) / 1e6)


def test_codex_cli_failure_raises(spawn):
	spawn.procs.append(Proc(json.dumps({'type': 'turn.failed', 'error': {'message': 'not signed in'}}).encode(), rc=1))
	with pytest.raises(ModelProviderError, match='not signed in'):
		inv(L.CodexCLI('gpt-5.5'))


def test_frontier_models_answer_only():
	assert {'opus55', 'sonnet55', 'gpt-astra'} <= L.ANSWER_ONLY and L.price_of('codex:gpt-6-astra') == (10.0, 50.0)


def test_reply_with_json_then_a_repeated_final_answer_parses(fake, monkeypatch):
	monkeypatch.setattr(L, 'routes', lambda name: [('anthropic', 'claude-opus-5-5')])
	fake.queue += [ok('{"answer": "6.6%", "confidence": 1}\n```\n\nFinal answer:\n\n{"answer": "6.6%", "confidence": 1}')]
	p, _, _ = asyncio.run(L.call('opus55', 'sys', 'u', Ans, retries=1))
	assert p.answer == '6.6%'


def test_a_hung_request_never_blocks_its_timeout(monkeypatch):
	import threading
	import time

	import httpx

	gate = threading.Event()

	class Hung:
		is_closed = False

		def __init__(self, timeout=None, **kw):
			pass

		def post(self, *a, **k):
			gate.wait(5)  # the server never answers (and never finishes closing)
			raise RuntimeError('late')

	monkeypatch.setattr(httpx, 'Client', Hung)
	monkeypatch.setattr(net, '_local', threading.local())
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'k')
	t0 = time.monotonic()
	with pytest.raises(asyncio.TimeoutError):
		asyncio.run(asyncio.wait_for(api().ainvoke([UserMessage(content='q')]), 0.2))
	gate.set()
	assert time.monotonic() - t0 < 2
