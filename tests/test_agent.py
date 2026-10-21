"""The browser agent without a framework: tools, trace, step budget, the MCP bridge for the CLI agents, the browser
client and the Modal teardown - all offline."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from conftest import FakeBrowser
from pydantic import BaseModel

import rl_env_use.agent as A
import rl_env_use.cloud as cloud
import rl_env_use.llm as L
from rl_env_use.browse import Browser, BrowserError, host_allowed
from rl_env_use.graders import grounded, max_repeats


class Out(BaseModel):
	answer: str
	source_url: str = ''


class Nav(FakeBrowser):
	"""A fake browser that follows navigate."""

	async def goto(self, url):
		self.at = url
		return {'url': url, 'title': 'Study'}


def tools(max_steps=5, extra=()):
	b = Nav('Enrollment (Actual) 28 participants. Primary outcome met.', 'https://clinicaltrials.gov/study/NCT1')
	return A.Tools([*A.browser_tools(b), *extra], b, max_steps, Out), b


def test_trace_marks_each_step_with_the_page_it_left_the_agent_on():
	ts, b = tools()
	run = lambda n, **kw: asyncio.run(ts.call(n, kw))  # noqa: E731
	text, err = run('navigate', url='https://clinicaltrials.gov/study/NCT2')
	assert not err and 'Opened https://clinicaltrials.gov/study/NCT2' in text and 'Enrollment' in text
	text, err = run('read_page', find='enrollment')
	assert not err and text.startswith('@0: Enrollment (Actual) 28') and 'read_page(start=<offset>)' in text
	assert 'No line' in run('read_page', find='withdrawn')[0] and 'No line' in run('read_page', find='enrollment', start=60)[0]
	assert run('done', answer='28', source_url=b.at) == ('Done. Stop now.', False) and ts.result == Out(answer='28', source_url=b.at)
	assert run('read_page') == ('Already done: stop now.', False)
	roles = [t['role'] for t in ts.trace]
	assert roles == ['assistant', 'tool_call', 'tool_result'] * 5 and ts.trace[0]['content'] == '[step 1 @ https://clinicaltrials.gov/study/NCT2]'
	assert ts.urls == ['https://clinicaltrials.gov/study/NCT2'] * 5 and ts.steps == 5
	assert grounded([{'url': 'https://clinicaltrials.gov/study/NCT2', 'value': '28'}], ts.trace) and max_repeats(ts.trace) == 1


def test_bad_calls_are_errors_the_model_reads_and_crashes_are_kept():
	@A.tool('A tool that crashes.')
	async def boom(x: int) -> str:
		raise ValueError('kaput')

	ts, _ = tools(extra=[boom])
	text, err = asyncio.run(ts.call('nope', {}))
	assert err and 'Unknown tool' in text
	text, err = asyncio.run(ts.call('navigate', {'url': 'clinicaltrials.gov'}))
	assert err and 'full http(s) URL' in text
	text, err = asyncio.run(ts.call('boom', {'x': 'not a number'}))
	assert err and 'Invalid arguments' in text and not ts.errors
	text, err = asyncio.run(ts.call('boom', {'x': 1}))
	assert err and ts.errors == ['ValueError: kaput'] and ts.trace[-1]['content'].startswith('ERROR: ')


def test_step_budget_asks_for_done_then_ends_the_run():
	ts, _ = tools(max_steps=2)
	for _ in range(2):
		assert not asyncio.run(ts.call('read_page', {}))[1]
	text, err = asyncio.run(ts.call('read_page', {}))
	assert err and 'call done now' in text and not ts.over
	for _ in range(A.STEP_GRACE):
		asyncio.run(ts.call('read_page', {}))
	assert ts.over  # a CLI agent that keeps going is stopped
	assert asyncio.run(ts.call('done', {'answer': 'x'}))[0] == 'Done. Stop now.'  # done is always accepted


def test_tool_schema_comes_from_the_signature():
	@A.tool('Record it.')
	async def record(claim: str, value: str = '', n: int = 3) -> str:
		return 'ok'

	s = record.schema()
	assert s['required'] == ['claim'] and set(s['properties']) == {'claim', 'value', 'n'} and 'title' not in s and s['properties']['n']['default'] == 3


def test_openai_loop_runs_tools_until_done(monkeypatch):
	replies = [
		(
			{
				'content': 'I will open the study.',
				'tool_calls': [{'id': 'c1', 'function': {'name': 'navigate', 'arguments': json.dumps({'url': 'https://clinicaltrials.gov/study/NCT9'})}}],
			},
			{'input_tokens': 10},
		),
		({'content': '', 'reasoning_content': 'it says 28'}, {'input_tokens': 5}),  # no call: nudged once
		({'content': '', 'tool_calls': [{'id': 'c2', 'function': {'name': 'done', 'arguments': '{"answer": "28"}'}}]}, {'input_tokens': 20}),
	]
	seen = []

	async def chat(self, msgs, tools=None, max_tokens=0):
		seen.append([dict(m) for m in msgs])
		msg, u = replies.pop(0)
		return msg, u, 'stop'

	monkeypatch.setattr(L.OpenAICompat, 'chat', chat)
	ts, _ = tools()
	usage, stop = asyncio.run(A.openai_loop('modal', 'Qwen/x', 'sys', 'task', ts))
	assert stop == 'done' and ts.result.answer == '28' and usage['input_tokens'] == 35
	assert seen[1][-1] == {'role': 'tool', 'tool_call_id': 'c1', 'content': seen[1][-1]['content']} and 'Opened' in seen[1][-1]['content']
	assert seen[2][-1]['content'].startswith('Continue with the tools')
	assert [t['content'] for t in ts.trace if t['role'] == 'assistant' and not t['content'].startswith('[step')] == ['I will open the study.', '']


def test_old_tool_results_are_trimmed():
	msgs = [{'role': 'tool', 'content': 'x' * 2000} for _ in range(6)]
	A.compact(msgs)
	assert [len(m['content']) < 2000 for m in msgs] == [True, True, False, False, False, False]


def test_the_mcp_bridge_relays_list_and_call_to_this_process():
	ts, _ = tools()

	async def go():
		async with A.Relay(ts) as relay:
			cmd, args = relay.command()
			proc = await asyncio.create_subprocess_exec(cmd, *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
			reqs = [
				{'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2025-06-18'}},
				{'jsonrpc': '2.0', 'method': 'notifications/initialized'},
				{'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'},
				{'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'read_page', 'arguments': {'find': 'primary'}}},
				{'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'done', 'arguments': {}}},
				{'jsonrpc': '2.0', 'id': 5, 'method': 'resources/list'},
			]
			out, _ = await asyncio.wait_for(proc.communicate(''.join(json.dumps(r) + '\n' for r in reqs).encode()), 20)
			return [json.loads(x) for x in out.decode().splitlines()]

	r = {x['id']: x for x in asyncio.run(go())}
	assert set(r) == {1, 2, 3, 4, 5} and r[1]['result']['capabilities'] == {'tools': {}}
	names = [t['name'] for t in r[2]['result']['tools']]
	assert names[-1] == 'done' and 'navigate' in names and r[2]['result']['tools'][-1]['inputSchema']['required'] == ['answer']
	assert r[3]['result']['content'][0]['text'].startswith('@0: Enrollment (Actual) 28 participants. Primary outcome met.') and not r[3]['result']['isError']
	assert r[4]['result']['isError'] and 'Invalid arguments' in r[4]['result']['content'][0]['text']
	assert r[5]['error']['code'] == -32601 and ts.steps == 2


def test_the_bridge_refuses_a_wrong_token():
	ts, _ = tools()

	async def go():
		async with A.Relay(ts) as relay:
			cmd, args = relay.command()
			args[-1] = 'forged'
			proc = await asyncio.create_subprocess_exec(cmd, *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
			req = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': 'read_page', 'arguments': {}}}
			out, _ = await asyncio.wait_for(proc.communicate((json.dumps(req) + '\n').encode()), 20)
			return json.loads(out)

	assert 'error' in asyncio.run(go()) and ts.steps == 0


def test_bridge_is_standard_library_only():
	src = (Path(A.__file__).with_name('bridge.py')).read_text()
	imports = {ln.split()[1] for ln in src.splitlines() if ln.startswith(('import ', 'from '))}
	assert imports <= {'json', 'socket', 'sys'}
	assert subprocess.run([sys.executable, str(Path(A.__file__).with_name('bridge.py')), '1', 't'], input=b'', capture_output=True, timeout=10).returncode == 0


def test_claude_agent_runs_bare_with_only_the_relay(monkeypatch):
	seen = {}

	async def cli_agent(argv, prompt, tools, timeout):
		seen.update(argv=argv, prompt=prompt)
		tools.result = Out(answer='a')
		res = {'type': 'result', 'subtype': 'success', 'total_cost_usd': 0.03, 'usage': {'input_tokens': 9, 'output_tokens': 1}}
		return (json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'Found it.'}]}}) + '\n' + json.dumps(res)).encode()

	monkeypatch.setattr(A, 'cli_agent', cli_agent)
	ts, _ = tools()
	usage, stop = asyncio.run(A.claude_loop('claude-sonnet-4-6', 'SYS', 'TASK', ts, 60))
	argv = seen['argv']
	mcp = json.loads(argv[argv.index('--mcp-config') + 1])['mcpServers']['rlenv']
	assert argv[argv.index('--tools') + 1] == '' and argv[argv.index('--allowedTools') + 1] == 'mcp__rlenv' and '--strict-mcp-config' in argv
	assert mcp['args'][0].endswith('bridge.py') and usage['reported_cost'] == 0.03 and stop == 'done' and ts.trace[-1]['content'] == 'Found it.'


def test_claude_agent_failing_before_any_step_is_infra(monkeypatch):
	async def cli_agent(argv, prompt, tools, timeout):
		return json.dumps({'type': 'result', 'subtype': 'error_during_execution', 'is_error': True, 'result': 'usage limit reached'}).encode()

	monkeypatch.setattr(A, 'cli_agent', cli_agent)
	with pytest.raises(RuntimeError, match='usage limit'):
		asyncio.run(A.claude_loop('m', 's', 't', tools()[0], 60))


def test_codex_agent_approves_only_its_relay_tools(monkeypatch):
	seen = {}

	async def cli_agent(argv, prompt, tools, timeout):
		seen['argv'] = argv
		return b'{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 1}}'

	monkeypatch.setattr(A, 'cli_agent', cli_agent)
	usage, stop = asyncio.run(A.codex_loop('gpt-6-astra', 's', 't', tools()[0], 60))
	argv = seen['argv']
	assert 'mcp_servers.rlenv.default_tools_approval_mode="approve"' in argv and '--ignore-user-config' in argv and argv[argv.index('-s') + 1] == 'read-only'
	assert usage['input_tokens'] == 5 and stop == 'stopped'


def test_run_agent_books_usage_even_when_the_run_crashes(monkeypatch):
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	monkeypatch.setattr(A, 'LEDGER', L.LEDGER)
	monkeypatch.setattr(A, 'resolve', lambda m: ('modal', 'Qwen/x'))

	async def boom(*a):
		raise RuntimeError('vLLM exited')

	monkeypatch.setattr(A, 'openai_loop', boom)
	with pytest.raises(RuntimeError):
		asyncio.run(A.run_agent('qwen-35b', 'task', Nav('x'), Out))
	assert 'modal:Qwen/x' in L.LEDGER.by_model


def test_host_allowed():
	assert host_allowed('https://www.sec.gov/a', ['sec.gov']) and host_allowed('https://efts.sec.gov/x', ['*.sec.gov'])
	assert not host_allowed('https://evil-sec.gov/', ['sec.gov']) and host_allowed('https://anything.org', None)


def test_browser_client_parses_values_and_raises_page_errors(monkeypatch):
	posts = []

	async def post(url, body, headers=None, timeout=0):
		posts.append((url, body, headers))
		if 'boom' in body:
			return NS(status_code=500, text='Error: Cannot navigate to invalid URL\n    at stack')
		return NS(status_code=200, text=json.dumps({'v': {'url': 'https://a.org/', 'title': 'A'}}))

	monkeypatch.setattr('rl_env_use.browse.net.post', post)
	b = Browser(['a.org'])
	b.url_, b.headers = 'https://sb.modal.host', {'Authorization': 'Bearer t'}
	assert asyncio.run(b.goto('https://a.org/x')) == {'url': 'https://a.org/', 'title': 'A'}
	url, body, headers = posts[-1]
	assert url == 'https://sb.modal.host/eval' and 'rl.goto(...["https://a.org/x"])' in body and headers['Authorization'] == 'Bearer t'
	with pytest.raises(BrowserError, match='invalid URL'):
		asyncio.run(b.js('boom'))
	with pytest.raises(BrowserError, match='outside the allowed domains'):
		asyncio.run(b.goto('https://b.org/'))


class Terminate:
	def __init__(self):
		self.n = 0

	def __call__(self):
		self.n += 1

	async def aio(self):
		self.n += 1


class FakeSandbox:
	def __init__(self, i):
		self.object_id, self.terminate = f'sb-{i}', Terminate()

	@property
	def terminated(self):
		return self.terminate.n


def test_every_container_is_stopped_and_booked_at_shutdown(monkeypatch):
	monkeypatch.setattr(L, 'LEDGER', L.Ledger())
	monkeypatch.setattr(cloud, 'LIVE', {})
	boxes = [FakeSandbox(i) for i in range(3)]
	for sb, label, rate in zip(boxes, ['browser', 'browser', 'llm Qwen/x'], [0.0001, 0.0001, 0.001]):
		cloud.LIVE[sb.object_id] = cloud.Live(sb, label, rate)
	cloud.LIVE['sb-2'].allocated = 1e9  # its requests already carried more than its life cost: nothing more to book
	asyncio.run(cloud.terminate(cloud.LIVE['sb-0']))
	assert boxes[0].terminated == 1 and 'sb-0' not in cloud.LIVE
	asyncio.run(cloud.terminate(cloud.Live(boxes[0], 'browser', 0.1)))  # already stopped: not stopped or booked twice
	assert boxes[0].terminated == 1
	asyncio.run(cloud.shutdown())
	assert [b.terminated for b in boxes] == [1, 1, 1] and cloud.LIVE == {}
	assert L.LEDGER.by_model['modal llm Qwen/x'] == {'reported_cost': 0.0} and L.LEDGER.by_model['modal browser']['reported_cost'] > 0


def test_the_atexit_sweep_stops_what_a_library_user_left_running(monkeypatch):
	monkeypatch.setattr(cloud, 'LIVE', {})
	sb = FakeSandbox(9)
	cloud.LIVE[sb.object_id] = cloud.Live(sb, 'browser', 0.0)
	cloud._sweep()
	assert sb.terminated == 1 and cloud.LIVE == {}


def test_a_browser_that_fails_to_start_is_stopped(monkeypatch):
	stopped = []

	async def browser_sandbox(proxy=None):
		return NS(sb=NS(object_id='x')), 'https://sb', 'tok'

	async def terminate(live):
		stopped.append(live)

	async def post(*a, **k):
		return NS(status_code=500, text='Error: no browser')

	monkeypatch.setattr(cloud, 'browser_sandbox', browser_sandbox)
	monkeypatch.setattr(cloud, 'terminate', terminate)
	monkeypatch.setattr('rl_env_use.browse.net.post', post)
	with pytest.raises(BrowserError):
		asyncio.run(Browser().start())
	assert len(stopped) == 1


def test_scout_mission_stops_its_browser_when_the_agent_crashes(monkeypatch):
	import rl_env_use.scout as S

	events = []

	class B(FakeBrowser):
		async def start(self):
			events.append('start')
			return self

		async def kill(self):
			events.append('kill')

	async def run_agent(*a, **k):
		raise RuntimeError('agent crashed')

	monkeypatch.setattr(S, 'browser', lambda *a, **k: B())
	monkeypatch.setattr(S, 'run_agent', run_agent)
	with pytest.raises(RuntimeError):
		asyncio.run(S.run_mission(S.Mission(angle='cases', instructions='i', start_url='https://a.org'), 't', {'facts': [], 'materials': []}, 'sonnet46', 5))
	assert events == ['start', 'kill']
