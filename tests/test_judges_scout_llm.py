import asyncio

import pytest
from conftest import FakeBrowser, mk_task, tool_runner

import rl_env_use.llm as L
import rl_env_use.scout as S
from rl_env_use.judges import FailureMode, Judge, agreement, calibrate, codify, load_judges, save_judges, trace_text

MODE = FailureMode(
	id='premature_stop',
	name='Premature stop',
	definition='Stops before reaching the answer page.',
	fail_if=['done called on a search page'],
	pass_if='answer read from the source page',
	example_ids=['a'],
)


def test_agreement():
	s = agreement([True, True, False, False, True], [True, False, False, False, True])
	assert s == {'n': 5, 'pos': 2, 'neg': 3, 'tpr': 1.0, 'tnr': 2 / 3, 'acc': 0.8}
	assert agreement([False], [False])['tpr'] is None


def test_calibrate_trusts_only_agreeing_consistent_judges():
	labeled = [(mk_task(i), [], 'x', i % 2 == 0) for i in range(20)]

	def judge(fn):
		j = Judge(MODE)
		j.run = fn
		return j

	async def perfect(t, tr, a):
		return {'score': 0.0 if int(t['id'][1:]) % 2 == 0 else 1.0}

	async def lenient(t, tr, a):
		return {'score': 1.0}

	good, bad = judge(perfect), judge(lenient)
	assert asyncio.run(calibrate(good, labeled))['trusted'] and good.stats['flip_rate'] == 0
	s = asyncio.run(calibrate(bad, labeled))
	assert s['tpr'] == 0 and not s['trusted']


def test_judges_persist_with_trust_filter(tmp_path):
	j1, j2 = Judge(MODE, stats={'trusted': True}), Judge(MODE.model_copy(update={'id': 'other'}), stats={'trusted': False})
	save_judges(tmp_path / 'j.json', [j1, j2])
	assert [j.id for j in load_judges(tmp_path / 'j.json', only_trusted=True)] == ['premature_stop']
	assert load_judges(tmp_path / 'missing.json') == []


def test_trace_text_truncates_and_renders_roles():
	tr = [{'role': 'system', 'content': 'T'}, {'role': 'tool_call', 'name': 'click', 'content': '{}'}, {'role': 'tool_result', 'content': 'x' * 5000}]
	s = trace_text(tr)
	assert s.startswith('TASK: T') and '-> click({})' in s and len(s) < 1700
	assert '[truncated]' in trace_text(tr * 100, limit=2000)


def test_codify_needs_failure_notes():
	with pytest.raises(ValueError):
		asyncio.run(codify([{'id': 'a', 'verdict': 'pass', 'notes': 'fine'}]))


def test_quote_in():
	assert S.quote_in('137 websites', 'Collected from 137\n websites spanning')
	assert not S.quote_in('138 websites', 'from 137 websites') and not S.quote_in('a', 'a')


def test_record_fact_rejects_hallucinated_quotes(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': []}
	b = FakeBrowser('Mind2Web: tasks collected from 137 websites spanning 31 domains.', 'https://arxiv.org/abs/2306.06070')
	tools = tool_runner(S.make_tools(kb, 'longtail', b))
	run = lambda **kw: tools('record_fact', **kw)  # noqa: E731
	assert run(claim='sites', value='140', quote='collected from 140 websites').error  # quote not on page
	assert run(claim='sites', value='140', quote='collected from 137 websites').error  # value not on page
	assert run(claim='updated', value='12 months ago', quote='12 months ago').error  # time-unstable
	assert run(claim='desc', value='x' * 81, quote='137 websites').error  # not atomic
	assert not run(claim='sites', value='137', quote='collected from 137 websites').error
	assert run(claim='sites', value='137', quote='137 websites').error  # quote too thin to show context
	assert 'Already recorded' in run(claim='sites again', value='137', quote='from 137 websites spanning').extracted_content
	assert [f['value'] for f in kb['facts']] == ['137'] and kb['facts'][0]['url'].endswith('2306.06070')

	b.at = 'https://duckduckgo.com/?q=mind2web'
	assert 'not evidence' in run(claim='sites', value='31', quote='spanning 31 domains here').error


def test_model_routing(monkeypatch):
	monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
	monkeypatch.setattr(L.shutil, 'which', lambda b: None)  # no local claude / codex CLI unless stated
	with pytest.raises(RuntimeError):
		L.resolve('haiku')
	monkeypatch.setenv('ANTHROPIC_API_KEY', 'x')
	assert L.resolve('haiku') == ('anthropic', 'claude-haiku-4-5')
	monkeypatch.setattr(L.shutil, 'which', lambda b: f'/usr/bin/{b}')
	assert L.resolve('haiku') == ('claude', 'claude-haiku-4-5')  # your Claude account first, the API key is the fallback
	assert L.resolve('claude-opus') == ('claude', 'opus')
	assert L.resolve('gpt') == ('codex', 'gpt-5.5')
	assert L.resolve('bu') == ('claude', 'claude-sonnet-4-6') and L.resolve('bu-mini') == ('claude', 'claude-haiku-4-5')


def test_cost():
	assert L.cost('anthropic/claude-sonnet-4-6', {'input_tokens': 1e6, 'output_tokens': 0}) == 3.0
	assert L.cost('anthropic/claude-sonnet-5.5', {'input_tokens': 1e6, 'output_tokens': 1e6}) == 12.0
	assert L.cost('claude-haiku-4-5', {'input_tokens': 1e6, 'cache_read_input_tokens': 1e6, 'output_tokens': 0}) == pytest.approx(0.1)
	assert L.cost('claude-cli:claude-opus-5-5', {'input_tokens': 5, 'reported_cost': 0.0085}) == 0.0085
	assert L.cost('mystery', {'input_tokens': 5}) is None
	assert L.cost('modal:Qwen/Qwen3.6-35B-A3B-FP8', {'input_tokens': 1e6, 'output_tokens': 1e6}) is None  # GPU time comes as reported_cost
	assert L.usage_dict(None) == {} and L.usage_dict({'input_tokens': 1}) == {'input_tokens': 1}


def test_untrusted_without_enough_labels():
	async def perfect(t, tr, a):
		return {'score': 0.0}

	j = Judge(MODE)
	j.run = perfect
	assert not asyncio.run(calibrate(j, [(mk_task(0), [], 'x', True)]))['trusted']  # n=1, tnr undefined


def test_look_measures_what_plain_http_sees(tmp_path, monkeypatch):
	import json

	import rl_env_use.pipeline as F

	kb = {
		'topic': 't',
		'facts': [
			{'id': 'f0', 'claim': 'c', 'value': '0.0012', 'quote': 'allele frequency 0.0012 in gnomAD', 'url': 'https://a.org/v1'},
			{'id': 'f1', 'claim': 'c', 'value': '7', 'quote': 'has 7 exons in total', 'url': 'https://b.org/g'},
			{'id': 'f2', 'claim': 'c', 'value': '3', 'quote': 'seen in 3 tissues here', 'url': 'https://c.org/x'},
		],
		'missions': [],
		'ideas': [],
		'sources': [],
	}
	(tmp_path / 'kb.json').write_text(json.dumps(kb))
	pages = {
		'https://a.org/v1': {'text': 'allele frequency 0.0012 in gnomAD', 'raw': '', 'blocked': False},
		'https://b.org/g': {'text': 'Loading...', 'raw': '{"note": "has 7 exons in total"}', 'blocked': True},
		'https://c.org/x': {'text': 'Loading...', 'raw': '', 'blocked': True},
	}

	async def fake(u):
		return pages[u]

	monkeypatch.setattr(F, 'http_capture', fake)
	out = asyncio.run(F.look(tmp_path))
	assert [f['http'] for f in json.loads((tmp_path / 'kb.json').read_text())['facts']] == ['visible', 'in_raw_html', 'blocked']
	assert '33% visible' in out and '33% unreachable without a browser' in out


def test_source_health_feeds_the_next_plan(monkeypatch):
	import asyncio

	import rl_env_use.scout as S

	kb = {
		'missions': [
			{'start_url': 'https://carol.ntsb.gov/', 'error': 'CancelledError'},
			{'start_url': 'https://carol.ntsb.gov/x', 'observations': ['CAROL is unstable for agents'], 'recorded': {'materials': 2}},
			{'start_url': 'https://www.ntsb.gov/a', 'recorded': {'materials': 3, 'outcomes': 3}},
		]
	}
	h = S.source_health(kb)
	assert (
		'carol.ntsb.gov: 2 missions (1 crashed) saved 2 materials and 0 outcomes - miner note: CAROL is unstable' in h
		and 'www.ntsb.gov: 1 missions (0 crashed) saved 3 materials and 3 outcomes' in h
	)
	seen = {}

	async def call(model, system, user, schema, **kw):
		seen.update(system=system, user=user)
		return S.Plan(missions=[S.Mission(angle='cases', instructions='i', start_url='https://www.ntsb.gov/a')]), {}, model

	monkeypatch.setattr(S, 'call', call)
	asyncio.run(S.plan('t', 1, 'm', ['cases'], ['https://carol.ntsb.gov/'], h))
	assert 'SOURCE HEALTH' in seen['user'] and 'move missions away' in seen['system']
	assert S.source_health({'missions': []}) == ''


def test_mission_tally_counts_what_was_saved(monkeypatch):
	import rl_env_use.scout as S

	page = (
		'ITEM 2. LIQUIDITY Cash fell to $4.1 million while the company burned $3.0 million per quarter. '
		+ 'Filler sentence about operations. ' * 12
		+ 'We may need to raise additional capital.'
	)

	from conftest import FakeBrowser, tool_runner

	kb, tally = {'facts': [], 'ideas': [], 'sources': []}, {}
	run = tool_runner(S.make_tools(kb, 'cases', FakeBrowser(page, 'https://www.sec.gov/a.htm'), tally=tally))
	run('record_material', case='acme', title='t', starts_with='ITEM 2. LIQUIDITY', ends_with='raise additional capital.')
	run('record_fact', claim='cash', value='$4.1 million', quote='Cash fell to $4.1 million while', case='acme')
	assert tally == {'materials': 1, 'outcomes': 1}


def test_planner_reads_the_source_pages(monkeypatch):
	import asyncio

	import rl_env_use.scout as S

	seen = {}

	async def page(url, *a, **k):
		return {'text': '{"studies": [{"nctId": "NCT07522918"}, {"nctId": "NCT06815562"}]}'}

	async def call(model, system, user, schema, **kw):
		seen.update(system=system, user=user)
		return S.Plan(missions=[S.Mission(angle='cases', instructions='i', start_url='https://clinicaltrials.gov/study/NCT07522918')]), {}, model

	monkeypatch.setattr(S, 'http_capture', page)
	monkeypatch.setattr(S, 'call', call)
	asyncio.run(S.plan('t', 1, 'm', ['cases'], ['https://clinicaltrials.gov/api/v2/studies']))
	assert 'SOURCE PAGES' in seen['user'] and 'NCT06815562' in seen['user'] and 'never get the same record' in seen['system']


def test_quote_may_skip_table_cells_with_an_ellipsis():
	from rl_env_use.browse import quote_in

	t = 'Overall Study STARTED 22 COMPLETED 22 NOT COMPLETED 0 Serious adverse events 0/22'
	assert quote_in('Started 22 ... Completed 22', t) and quote_in('Started 22 … Serious adverse events 0/22', t)
	assert not quote_in('Completed 22 ... Started 22', t) and not quote_in('Started 22 ... Completed 23', t)
	assert not quote_in('Started 22 ... Serious adverse events 0/22', 'Started 22 ' + 'x ' * 200 + 'Serious adverse events 0/22')  # pieces must be near each other


def test_a_kill_cancels_the_work_so_cleanup_runs():
	import asyncio
	import os
	import signal
	import threading

	from rl_env_use.__main__ import run

	done = []

	async def agent():
		try:
			await asyncio.sleep(10)
		finally:
			done.append('closed')  # where an agent stops its browser

	async def both():
		threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
		await asyncio.gather(agent(), agent())

	with pytest.raises(KeyboardInterrupt):
		run(both())
	assert done == ['closed', 'closed']
