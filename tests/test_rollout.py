import asyncio
import json

import pytest
from conftest import jl, mk_task

import rl_env_use.rollout as R


def test_public_hides_gold():
	pub = R.public(mk_task(0))
	assert set(pub) == {'id', 'prompt', 'start_url', 'allowed_domains'}


@pytest.fixture
def fake_run(monkeypatch):
	calls = []

	async def run_case(t, rep, model, vdir, flow, max_steps, headless, judges=None, bu_judge=False, extend_system=None):
		calls.append((t['id'], rep))
		if t['id'] == 'boom':
			raise RuntimeError('browser crashed')
		if t['id'] == 'slow':
			await asyncio.sleep(5)
		return [{'role': 'system', 'content': 'x'}], {'prompt_id': t['id'], 'rep': rep, 'status': 'ok', 'grade': {'correct': 1.0}}

	monkeypatch.setattr(R, 'run_case', run_case)
	return calls


def test_rollout_writes_contract_and_resumes(tmp_path, fake_run):
	tasks = [mk_task(0), {**mk_task(1), 'id': 'boom'}, {**mk_task(2), 'id': 'slow'}]
	v = asyncio.run(R.rollout(tasks, tmp_path, reps=2, timeout_s=0.5))
	rows, errs = jl(v / 'results.jsonl'), jl(v / 'errors.jsonl')
	assert sorted((r['prompt_id'], r['rep']) for r in rows) == [('t0', 0), ('t0', 1)]
	assert {(e['prompt_id'], e['failure_class']) for e in errs} == {('boom', 'harness_error'), ('slow', 'timeout')}
	assert json.loads((v / 'traces' / 't0_rep1.json').read_text())[0]['role'] == 'system'
	fake_run.clear()
	asyncio.run(R.rollout(tasks, tmp_path, reps=2, timeout_s=0.5))
	assert ('t0', 0) not in fake_run and ('boom', 0) in fake_run  # finished cases skipped, infra failures retried
	assert len(jl(v / 'results.jsonl')) == 2


def test_rollout_survives_cancelled_error(tmp_path, monkeypatch):
	async def run_case(t, *a, **k):
		if t['id'] == 't1':
			raise asyncio.CancelledError('DOMWatchdog interrupted')
		return [], {'prompt_id': t['id'], 'rep': 0, 'status': 'ok', 'grade': {'correct': 1.0}}

	monkeypatch.setattr(R, 'run_case', run_case)
	v = asyncio.run(R.rollout([mk_task(0), mk_task(1)], tmp_path))
	assert [r['prompt_id'] for r in jl(v / 'results.jsonl')] == ['t0'] and jl(v / 'errors.jsonl')[0]['failure_class'] == 'harness_error'


def test_runner_retries_when_browser_pool_is_full(tmp_path, monkeypatch):
	import rl_env_use.runner as RN

	calls = []

	async def nosleep(_):
		pass

	async def run_one(c, rep, vdir):
		calls.append(c['id'])
		if c['id'] == 'busy' and calls.count('busy') < 2:
			raise RuntimeError('Failed to create cloud browser: HTTP 429 - Too many concurrent active sessions')
		if c['id'] == 'crash':
			raise RuntimeError('boom')
		return [], {'prompt_id': c['id'], 'rep': rep, 'status': 'ok'}

	monkeypatch.setattr(RN.asyncio, 'sleep', nosleep)
	v = asyncio.run(RN.run_eval([{'id': 'busy'}, {'id': 'crash'}], run_one, tmp_path, concurrency=1))
	assert calls.count('busy') == 2 and calls.count('crash') == 1
	assert [r['prompt_id'] for r in jl(v / 'results.jsonl')] == ['busy'] and [r['prompt_id'] for r in jl(v / 'errors.jsonl')] == ['crash']
