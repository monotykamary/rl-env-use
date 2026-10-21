"""Difficulty panel: several labs solve each kept case task; all-pass is saturated, all-fail/shared-wrong need adjudication."""

import asyncio
import copy
import json
from collections import Counter

import pytest
from conftest import jl, mk_task, write_jl

import rl_env_use.pipeline as P
import rl_env_use.rollout as R
import rl_env_use.tasks as T
import rl_env_use.verify as V

CTX = '### Acme 10-K\nRevenue was 1,000 in 2023 and 500 in 2022.'
OK = {
	'structural': [],
	'leakage': False,
	'live': True,
	'closed_book': 'skipped',
	'critic': {'unambiguous': True, 'gold_supported': True, 'agrees': True, 'answer_in_context': False, 'well_posed': True, 'issues': ''},
}
MODELS = ['haiku', 'sonnet46', 'gpt6', 'gemini']  # a four-lab panel for these tests


def ct(i=0, answer='1500', answer_type='number', kind='derive', checks=None, **kw):
	t = mk_task(
		i,
		answer=answer,
		answer_type=answer_type,
		context=CTX,
		kind=kind,
		case=f'c{i}',
		material_ids=['m0'],
		start_url='',
		fact_ids=['f0'],
		prompt=f'What was total revenue in 2022-2023 for case {i}?',
		expression='1000 + 500' if kind == 'derive' else '',
		checks=copy.deepcopy(checks or OK),
	)
	return t | kw


def fake(answers, adj=None, log=None):
	"""answers[model]: str | Exception | list (one per material call, in order) | dict(mat=..., blind=...)."""
	log = [] if log is None else log
	seen = {}

	async def call(model, system, user, schema, **kw):
		log.append({'model': model, 'system': system, 'user': user, 'schema': schema, 'kw': kw})
		if system.startswith('You adjudicate'):
			if isinstance(adj, Exception):
				raise adj
			return V.Adjudication(**adj), {}, model
		mode = 'blind' if '(materials withheld)' in user else 'mat'
		a = answers[model]
		a = a[mode] if isinstance(a, dict) else a
		if isinstance(a, list):
			n = seen[model, mode] = seen.get((model, mode), -1) + 1
			a = a[n]
		if isinstance(a, Exception):
			raise a
		return R.Reasoned(reasoning='work', answer=a), {}, model

	return call, log


def run(monkeypatch, tasks, answers, adj=None, **kw):
	call, log = fake(answers, adj)
	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)  # panel answers go through rollout.solve_case
	kw = {'models': MODELS, 'adjudicators': ['gpt6', 'gemini', 'sonnet46']} | kw
	asyncio.run(V.panel(tasks, {'f0': {'quote': 'total revenue was 1,500', 'url': 'u'}}, **kw))
	return log


adj_calls = lambda log: [c for c in log if c['system'].startswith('You adjudicate')]  # noqa: E731
CONFIRM = {'gold_correct': True, 'ambiguous': False, 'panel_error': 'models summed the wrong years'}


def test_all_right_is_saturated_and_no_adjudication(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {m: '1,500' for m in MODELS})
	p = t['checks']['panel']
	assert p['verdict'] == 'saturated' and p['scores'] == {m: 1.0 for m in MODELS}
	assert len(log) == 2 * len(MODELS) and not adj_calls(log) and all(c['schema'] is R.Reasoned and c['kw'] == {'temperature': None} for c in log)
	assert all('(materials withheld)' not in c['user'] and CTX in c['user'] for c in log)  # derive tasks are never solved blind
	assert p['answers'] == {m: ['1,500', '1,500'] for m in MODELS} and p['blind'] == {}
	assert not T.keep(t)


def test_policy_never_sees_gold(monkeypatch):
	t = ct(answer='98765')
	log = run(monkeypatch, [t], {m: '1' for m in MODELS}, CONFIRM)
	assert all('98765' not in c['user'] for c in log if not c['system'].startswith('You adjudicate'))
	assert '98765' in adj_calls(log)[0]['user']


def test_mixed_is_kept_without_adjudication(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '1000', 'sonnet46': ['1500', '900'], 'gpt6': '1500', 'gemini': '1500'})
	p = t['checks']['panel']
	assert p['verdict'] == 'keep' and p['all_fail'] is False and not adj_calls(log) and 'adjudication' not in p
	assert p['scores'] == {'haiku': 0.0, 'sonnet46': 0.5, 'gpt6': 1.0, 'gemini': 1.0}
	assert T.keep(t)


def test_single_model_repeating_a_wrong_answer_is_not_shared(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '2000', 'sonnet46': '1500', 'gpt6': '1500', 'gemini': '1500'})
	assert t['checks']['panel']['verdict'] == 'keep' and not adj_calls(log)


def test_distinct_wrong_answers_are_not_shared(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '2000', 'sonnet46': '1000', 'gpt6': '1500', 'gemini': '1500'})
	assert t['checks']['panel']['verdict'] == 'keep' and not adj_calls(log)


def test_all_fail_confirmed_is_kept_as_all_fail(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '1000', 'sonnet46': '1000', 'gpt6': '500', 'gemini': '1000'}, CONFIRM)
	p = t['checks']['panel']
	assert p['verdict'] == 'keep' and p['all_fail'] is True and p['adjudication'] == CONFIRM | {'model': 'gpt6'}
	[a] = adj_calls(log)
	assert a['model'] == 'gpt6' and a['schema'] is V.Adjudication
	assert CTX in a['user'] and 'GOLD: 1500' in a['user'] and '1000 + 500' in a['user'] and 'total revenue was 1,500' in a['user']
	assert "MODELS' ANSWER: 1000 (3 models)" in a['user'] and not any(m in a['user'] for m in MODELS)  # the modal wrong answer and how many gave it, never who
	assert "THE DRAFTER'S COMPUTATION (arithmetic checked by code; its inputs and formula are NOT verified): 1000 + 500" in a['user']
	assert T.keep(t)


@pytest.mark.parametrize('verdict', [{'gold_correct': True, 'ambiguous': True}, {'gold_correct': False, 'ambiguous': False}, {'gold_correct': False, 'ambiguous': True}])
def test_all_fail_disputed(monkeypatch, verdict):
	t = ct()
	run(monkeypatch, [t], {m: '1000' for m in MODELS}, verdict | {'panel_error': 'x'})
	assert t['checks']['panel']['verdict'] == 'gold_disputed' and t['checks']['panel']['all_fail'] is True and not T.keep(t)


def test_shared_wrong_number_across_formats_is_adjudicated(monkeypatch):
	t = ct()
	log = run(
		monkeypatch, [t], {'haiku': '1,234', 'sonnet46': '1500', 'gpt6': '1500', 'gemini': '1234.0'}, {'gold_correct': False, 'ambiguous': False, 'panel_error': 'gold wrong'}
	)
	p = t['checks']['panel']
	assert p['all_fail'] is False and p['verdict'] == 'gold_disputed'
	[a] = adj_calls(log)
	assert "MODELS' ANSWER: 1234 (2 models)" in a['user'] and a['model'] == 'gpt6'
	assert V.answer_key(t, '1,234') == V.answer_key(t, '1234.0') == V.answer_key(t, '1234') != V.answer_key(t, '1235')


def test_shared_wrong_text_answer_normalized(monkeypatch):
	t = ct(answer='Acme Corp', answer_type='exact', expression='')
	log = run(monkeypatch, [t], {'haiku': 'Beta Inc.', 'sonnet46': 'Acme Corp', 'gpt6': 'beta inc', 'gemini': 'Acme Corp'}, CONFIRM)
	assert t['checks']['panel']['verdict'] == 'keep' and len(adj_calls(log)) == 1
	assert 'DERIVATION:' in adj_calls(log)[0]['user']  # no expression: the derivation is shown instead


def test_adjudicator_error_is_unmeasured_and_retried(monkeypatch):
	t = ct()
	run(monkeypatch, [t], {m: '1000' for m in MODELS}, RuntimeError('overloaded'))
	p = t['checks']['panel']
	assert p['verdict'] == 'unmeasured' and 'overloaded' in p['error'] and P.panel_todo(t) and not T.keep(t)


def test_errors_from_all_but_one_model_unmeasured(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': RuntimeError('refused'), 'sonnet46': RuntimeError('500'), 'gpt6': ValueError('refusal'), 'gemini': '1500'})
	p = t['checks']['panel']
	assert p['verdict'] == 'unmeasured' and p['scores'] == {'gemini': 1.0} and p['answers']['haiku'] == [None, None]
	assert not adj_calls(log) and P.panel_todo(t) and not T.keep(t)


def test_errored_models_are_not_scored_as_wrong(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': RuntimeError('refused'), 'sonnet46': [RuntimeError('x'), '1500'], 'gpt6': '1500', 'gemini': '1500'})
	p = t['checks']['panel']
	assert p['verdict'] == 'saturated' and 'haiku' not in p['scores'] and p['scores']['sonnet46'] == 1.0 and not adj_calls(log)


def prob_task(i=0, answer='yes'):
	return ct(i, answer=answer, answer_type='prob', kind='predict', prompt=f'Will case {i} file for bankruptcy within a year? Answer with the probability (0 to 1) that it did.')


def test_prob_shift_above_threshold_keeps(monkeypatch):
	t = prob_task()
	log = run(monkeypatch, [t], {m: {'mat': '0.7', 'blind': '0.5'} for m in MODELS})
	p = t['checks']['panel']
	assert len(log) == 3 * len(MODELS) and sum('(materials withheld)' in c['user'] for c in log) == len(MODELS)
	assert p['blind'] == {m: ['0.5'] for m in MODELS} and p['shift'] == pytest.approx(0.2) and p['blind_score'] == pytest.approx(0.75) and p['verdict'] == 'keep'
	assert 'lift' not in p and 'all_fail' not in p and not adj_calls(log) and T.keep(t)


def test_prob_shift_is_outcome_blind(monkeypatch):
	"""Materials that move the panel AWAY from the outcome (a genuine surprise) still count: the shift ignores the gold."""
	t = prob_task(answer='no')
	run(monkeypatch, [t], {m: {'mat': '0.7', 'blind': '0.5'} for m in MODELS})
	p = t['checks']['panel']
	assert p['shift'] == pytest.approx(0.2) and p['scores'] == {m: pytest.approx(0.51) for m in MODELS} and p['verdict'] == 'keep' and T.keep(t)


def test_prob_shift_is_mean_over_models(monkeypatch):
	t = prob_task()
	run(
		monkeypatch,
		[t],
		{
			'haiku': {'mat': '0.9', 'blind': '0.5'},
			'sonnet46': {'mat': ['0.6', '0.8'], 'blind': '0.5'},
			'gpt6': {'mat': '0.5', 'blind': '0.5'},
			'gemini': {'mat': '60%', 'blind': 'no'},
		},
	)
	assert t['checks']['panel']['shift'] == pytest.approx((0.4 + 0.2 + 0 + 0.6) / 4)


def test_prob_shift_skips_models_without_a_parseable_pair(monkeypatch):
	t = prob_task()
	run(
		monkeypatch,
		[t],
		{
			'haiku': {'mat': '0.9', 'blind': 'probably'},
			'sonnet46': {'mat': '0.7', 'blind': RuntimeError('x')},
			'gpt6': {'mat': '0.6', 'blind': '0.5'},
			'gemini': {'mat': '0.6', 'blind': '0.5'},
		},
	)
	p = t['checks']['panel']
	assert p['shift'] == pytest.approx(0.1) and p['blind_score'] == pytest.approx((0 + 0.75 + 0.75) / 3) and p['verdict'] == 'keep'


def test_prob_shift_below_threshold_is_no_shift(monkeypatch):
	t = prob_task(answer='no')
	run(monkeypatch, [t], {m: {'mat': '0.3', 'blind': '0.35'} for m in MODELS})
	p = t['checks']['panel']
	assert 0 < p['shift'] < V.SHIFT and p['blind_score'] < V.MEMORIZED and p['verdict'] == 'no_shift' and not T.keep(t)


def test_prob_memorized_outcome(monkeypatch):
	t = prob_task()
	run(monkeypatch, [t], {m: '0.99' for m in MODELS})
	p = t['checks']['panel']
	assert p['shift'] == 0 and p['blind_score'] >= V.MEMORIZED and p['verdict'] == 'memorized' and not T.keep(t)


def test_prob_memorized_wins_over_a_shift(monkeypatch):
	t = prob_task()
	run(monkeypatch, [t], {m: {'mat': '0.6', 'blind': 'yes'} for m in MODELS})
	p = t['checks']['panel']
	assert p['shift'] == pytest.approx(0.4) and p['blind_score'] == 1.0 and p['verdict'] == 'memorized'


def test_prob_saturated(monkeypatch):
	t = prob_task()
	run(monkeypatch, [t], {m: {'mat': '95%', 'blind': '0.5'} for m in MODELS})
	p = t['checks']['panel']
	assert p['scores'] == {m: pytest.approx(0.9975) for m in MODELS} and p['shift'] > V.SHIFT and p['verdict'] == 'saturated'


def test_prob_blind_all_errors_is_unmeasured(monkeypatch):
	t = prob_task()
	run(monkeypatch, [t], {m: {'mat': '0.6', 'blind': RuntimeError('x')} for m in MODELS})
	p = t['checks']['panel']
	assert p['shift'] is None and p['blind_score'] is None and p['verdict'] == 'unmeasured' and not T.keep(t)


def test_prob_task_without_panel_is_not_kept():
	assert not T.keep(prob_task())


def stage_one(tmp_path, monkeypatch, t, answers):
	env = tmp_path / 'env'
	write_jl(env / 'tasks.jsonl', [t])
	(env / 'kb.json').write_text(json.dumps({'facts': [{'id': 'f0', 'quote': 'total revenue was 1,500', 'url': 'u'}]}))
	call, log = fake(answers, CONFIRM)
	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)
	asyncio.run(P.panel_stage(env, {'panel': MODELS, 'adjudicators': ['gpt6', 'gemini', 'sonnet46']}))
	return jl(env / 'tasks.jsonl')[0], log


def test_panel_stage_measures_forecasts(tmp_path, monkeypatch):
	t, log = stage_one(tmp_path, monkeypatch, prob_task(), {m: {'mat': '0.7', 'blind': '0.5'} for m in MODELS})
	assert log and t['checks']['panel']['verdict'] == 'keep' and T.keep(t)


def test_panel_stage_measures_by_code_derive_with_critic_disagreement(tmp_path, monkeypatch):
	t, log = stage_one(tmp_path, monkeypatch, ct(checks=OK | {'critic': OK['critic'] | {'agrees': False}}), {'haiku': '1000', 'sonnet46': '1500', 'gpt6': '1500', 'gemini': '900'})
	assert log and t['checks']['panel']['verdict'] == 'keep' and T.keep(t)


def test_panel_lift_over_kept_forecasts():
	a, b, c, d = prob_task(0), prob_task(1), prob_task(2), ct(3)
	a['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 0.9, 'gpt6': 0.7}, 'blind_score': 0.75}  # +0.05
	b['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 0.5}, 'blind_score': 0.6}  # -0.1
	c['checks']['panel'] = {'verdict': 'no_shift', 'scores': {'haiku': 1.0}, 'blind_score': 0.0}  # not kept: ignored
	d['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 1.0}}  # not a forecast
	assert P.panel_lift([a, b, c, d]) == pytest.approx(-0.025) and P.panel_lift([c, d]) is None


def test_panel_custom_models_and_reps(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'a': '1500', 'b': '1000'}, models=['a', 'b'], k=3)
	assert len(log) == 6 and t['checks']['panel']['scores'] == {'a': 1.0, 'b': 0.0} and t['checks']['panel']['verdict'] == 'keep'


def kept_set(n, hard_ids):
	ts = [ct(i, id=f't{i:02d}') for i in range(n)]
	for t in ts:
		t['checks']['panel'] = {'verdict': 'keep', 'all_fail': t['id'] in hard_ids, 'scores': {}}
	return ts


def test_cap_all_fail_caps_by_id_order():
	ts = kept_set(20, {'t03', 't07', 't01', 't15', 't12', 't09'})
	ts.append(ct(99, id='t00', checks=OK | {'leakage': True}))  # not kept: neither counted nor capped
	ts[-1]['checks']['panel'] = {'verdict': 'keep', 'all_fail': True}
	assert V.cap_all_fail(ts) == 3  # 14 kept non-all-fail: floor(0.25 * 14) = 3 all-fail allowed
	assert sorted(t['id'] for t in ts if t['checks']['panel']['verdict'] == 'capped') == ['t09', 't12', 't15']
	assert ts[-1]['checks']['panel']['verdict'] == 'keep'
	assert sum(T.keep(t) for t in ts) == 17 and sum(T.keep(t) and t['checks']['panel']['all_fail'] for t in ts) == 3


def test_cap_all_fail_deterministic_under_shuffle():
	a, b = kept_set(15, {'t02', 't04', 't06', 't08', 't10'}), kept_set(15, {'t02', 't04', 't06', 't08', 't10'})
	b.reverse()
	assert V.cap_all_fail(a) == V.cap_all_fail(b) == 3
	cap = lambda ts: sorted(t['id'] for t in ts if t['checks']['panel']['verdict'] == 'capped')  # noqa: E731
	assert cap(a) == cap(b) == ['t06', 't08', 't10']


def test_cap_all_fail_under_cap_and_custom_cap():
	assert V.cap_all_fail(kept_set(10, {'t01', 't02'})) == 0
	assert V.cap_all_fail(kept_set(10, {'t01', 't02'}), cap=0.0) == 2


verdicts = lambda ts: [t['checks']['panel']['verdict'] for t in ts]  # noqa: E731


@pytest.mark.parametrize('n,hard', [(10, 3), (20, 6), (7, 7), (30, 2), (12, 5)])
def test_cap_all_fail_meets_its_cap_and_is_idempotent(n, hard):
	ts = kept_set(n, {f't{i:02d}' for i in range(hard)})
	first = V.cap_all_fail(ts)
	v = verdicts(ts)
	kept = [t for t in ts if T.keep(t)]
	assert sum(t['checks']['panel']['all_fail'] for t in kept) <= 0.2 * len(kept) + 1e-9
	assert V.cap_all_fail(ts) == first and verdicts(ts) == v


def test_cap_all_fail_readmits_when_env_grows():
	ts = kept_set(10, {'t00', 't01', 't02', 't03'})
	assert V.cap_all_fail(ts) == 3 and verdicts(ts)[:4] == ['keep', 'capped', 'capped', 'capped']
	ts += [ct(i, id=f't{i:02d}') for i in range(10, 16)]
	for t in ts[10:]:
		t['checks']['panel'] = {'verdict': 'keep', 'all_fail': False, 'scores': {}}
	assert V.cap_all_fail(ts) == 1 and verdicts(ts)[:4] == ['keep', 'keep', 'keep', 'capped']  # 12 others: floor(3) allowed


def test_cap_all_fail_ignores_other_verdicts():
	ts = kept_set(8, {'t00', 't01'})
	ts[0]['checks']['panel']['verdict'] = 'gold_disputed'
	assert V.cap_all_fail(ts) == 0 and verdicts(ts)[:2] == ['gold_disputed', 'keep']  # 6 others: floor(1.5) = 1 allowed


def test_panel_todo():
	t = ct()
	assert P.panel_todo(t)
	t['checks']['panel'] = {'verdict': 'unmeasured'}
	assert P.panel_todo(t)
	for v in ('keep', 'saturated', 'gold_disputed', 'no_shift', 'memorized', 'capped'):
		t['checks']['panel'] = {'verdict': v}
		assert not P.panel_todo(t)
	assert not P.panel_todo(ct(checks=OK | {'leakage': True}))
	assert not P.panel_todo(ct(checks=OK | {'critic': OK['critic'] | {'unambiguous': False}}))


def test_panel_means():
	a, b, c = ct(0), ct(1, kind='predict'), ct(2, checks=OK | {'live': False})
	a['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 0.0, 'gpt6': 1.0}}
	b['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 0.5, 'gpt6': 0.5}}
	c['checks']['panel'] = {'verdict': 'keep', 'scores': {'haiku': 1.0, 'gpt6': 1.0}}  # fails an earlier gate: ignored
	d = ct(3)
	d['checks']['panel'] = {'verdict': 'saturated', 'scores': {'haiku': 1.0, 'gpt6': 1.0}}  # not kept: ignored
	ts = [a, b, c, d, ct(4)]  # ct(4) kept but never measured
	assert P.panel_means(ts) == {'haiku': 0.25, 'gpt6': 0.75}
	assert P.panel_means(ts, 'derive') == {'haiku': 0.0, 'gpt6': 1.0}
	assert P.panel_means(ts, 'nothing') == {}
	assert P.panel_means(ts, 'derive', kept=False) == {'haiku': pytest.approx(0.667), 'gpt6': 1.0}  # every measured task, saturated d included
	assert P.panel_means([ct(4)], kept=False) == {}


def test_panel_stage_writes_verdicts_and_skips_gated(tmp_path, monkeypatch):
	env = tmp_path / 'env'
	done = ct(4)
	done['checks']['panel'] = {'verdict': 'keep', 'scores': {}, 'all_fail': False, 'models': ['a', 'b']}  # measured by this panel already
	browse = mk_task(5)  # no context: browse tasks are never paneled
	retry = ct(6)
	retry['checks']['panel'] = {'verdict': 'unmeasured'}
	tasks = [ct(0), ct(1), ct(2, checks=OK | {'structural': ['bad']}), ct(3, checks=OK | {'closed_book': {'haiku': True}}), done, browse, retry]
	write_jl(env / 'tasks.jsonl', tasks)
	(env / 'kb.json').write_text(json.dumps({'facts': [{'id': 'f0', 'quote': 'total revenue was 1,500', 'url': 'u'}]}))
	asked = []

	async def call(model, system, user, schema, **kw):
		asked.append(user)
		return R.Reasoned(reasoning='', answer='1500' if 'case 0' in user or model == 'b' else '9'), {}, model

	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)
	out = asyncio.run(P.panel_stage(env, {'panel': ['a', 'b'], 'panel_reps': 1}))
	got = {t['id']: t['checks'].get('panel', {}).get('verdict') for t in jl(env / 'tasks.jsonl')}
	assert got == {'t0': 'saturated', 't1': 'keep', 't2': None, 't3': None, 't4': 'keep', 't5': None, 't6': 'keep'}
	assert len(asked) == 3 * 2 and all('case 2' not in u and 'case 3' not in u and 'case 4' not in u for u in asked)
	assert out.startswith('panel: 3 tasks measured on a, b') and "'saturated': 1" in out and "'keep': 2" in out


def test_panel_stage_default_models_and_cap(tmp_path, monkeypatch):
	env = tmp_path / 'env'
	write_jl(env / 'tasks.jsonl', [ct(i, id=f't{i:02d}') for i in range(5)])
	(env / 'kb.json').write_text(json.dumps({'facts': []}))
	call, log = fake({m: '1000' for m in V.PANEL}, CONFIRM)  # every task all-fail, confirmed
	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)
	out = asyncio.run(P.panel_stage(env, {}))
	v = [t['checks']['panel']['verdict'] for t in jl(env / 'tasks.jsonl')]
	assert v == ['capped'] * 5 and '5 all-fail tasks over the cap' in out  # no non-all-fail task kept: none allowed
	assert V.PANEL == ['opus55', 'gpt-astra'] and {c['model'] for c in log} - {'gpt-astra'} == {'opus55'}
	assert len(adj_calls(log)) == 5 and {c['model'] for c in adj_calls(log)} == {'gpt-astra'}  # the first default adjudicator; both gave the modal answer
	assert all(
		('Think it through' in c['user']) == (c['model'] in ('opus55', 'gpt-astra')) for c in log if not c['system'].startswith('You adjudicate')
	)  # closed frontier models answer only


def test_panel_stage_passes_spec_adjudicators(tmp_path, monkeypatch):
	env = tmp_path / 'env'
	write_jl(env / 'tasks.jsonl', [ct(0)])
	(env / 'kb.json').write_text(json.dumps({'facts': []}))
	call, log = fake({m: '1000' for m in MODELS} | {'judge': '1000'}, CONFIRM)
	monkeypatch.setattr(V, 'call', call)
	monkeypatch.setattr(R, 'call', call)
	asyncio.run(P.panel_stage(env, {'panel': MODELS, 'adjudicators': ['gpt6', 'judge']}))
	assert [c['model'] for c in adj_calls(log)] == ['judge']


def test_adjudicator_skips_models_that_gave_the_modal_answer(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '1500', 'sonnet46': '1500', 'gpt6': '1000', 'gemini': '1000'}, CONFIRM)
	assert [c['model'] for c in adj_calls(log)] == ['sonnet46'] and t['checks']['panel']['adjudication']['model'] == 'sonnet46'


def test_adjudicator_falls_back_to_first_when_all_gave_it(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {'haiku': '1500', 'sonnet46': '1000', 'gpt6': '1000', 'gemini': '1000'}, CONFIRM)
	assert [c['model'] for c in adj_calls(log)] == ['gpt6']


def test_adjudicators_param(monkeypatch):
	t = ct()
	log = run(monkeypatch, [t], {m: '1000' for m in MODELS} | {'judge': '1000'}, CONFIRM, adjudicators=['haiku', 'judge'])
	assert [c['model'] for c in adj_calls(log)] == ['judge'] and V.ADJUDICATORS == ['gpt-astra', 'opus55']


def test_answers_stating_several_values_mark_a_multi_value_prompt(monkeypatch):
	t = ct()  # gold 1500
	run(monkeypatch, [t], {'haiku': '1500', 'sonnet46': '1500', 'gpt6': '1,000 and 500, total 1500', 'gemini': '1500'})
	p = t['checks']['panel']
	assert p['verdict'] == 'multi_value' and p['format_misses'] and not T.keep(t)
	assert V.format_misses(t, ['1500', '1,400, 1,600', None]) == []  # a hedge between two wrong values is a capability miss
	assert V.format_misses(ct(answer='yes', answer_type='prob'), ['0.7, 0.8']) == []


def test_a_new_panel_re_measures_tasks_measured_by_another():
	old = ct(0, checks=OK | {'panel': {'verdict': 'keep', 'scores': {'haiku': 0.0}}})  # legacy: no models recorded
	same = ct(1, checks=OK | {'panel': {'verdict': 'saturated', 'models': ['opus55', 'gpt-astra']}})
	assert P.panel_todo(old, V.PANEL) and not P.panel_todo(same, V.PANEL) and not P.panel_todo(old)


def test_rescore_applies_grader_fixes_to_stored_panel_answers():
	t = ct(checks=OK | {'panel': {'verdict': 'keep', 'models': MODELS, 'answers': {'haiku': ['1500 (1,000 + 500)'], 'gpt6': ['1500'], 'gemini': ['1,500']}, 'scores': {}}})
	V.rescore_panel(t)
	assert t['checks']['panel']['verdict'] == 'saturated' and t['checks']['panel']['scores']['haiku'] == 1.0
	d = ct(checks=OK | {'panel': {'verdict': 'gold_disputed', 'answers': {'haiku': ['1500'], 'gpt6': ['1500']}}})
	V.rescore_panel(d)
	assert d['checks']['panel']['verdict'] == 'gold_disputed'  # an adjudicated verdict is not re-opened


def fc(i, ans, case=None, verdict='keep'):
	return ct(i, answer=ans, answer_type='prob', kind='predict', checks=OK | {'panel': {'verdict': verdict, 'scores': {'a': 0.7}}}) | ({'case': case} if case else {})


def test_balanced_caps_the_majority_outcome_at_60_percent_from_the_busiest_cases():
	ts = [fc(i, 'yes', case='big' if i < 4 else f'c{i}') for i in range(8)] + [fc(8, 'no'), fc(9, 'no'), ct(10)]
	out = T.balanced(ts)
	f = [t for t in out if t['answer_type'] == 'prob']
	assert sum(t['answer'] == 'yes' for t in f) == 3 and len(f) == 5 and any(t['id'] == 't10' for t in out)  # 2 no -> 3 yes; derive untouched
	assert all(t.get('case') != 'big' for t in f if t['answer'] == 'yes')  # dropped from the case holding most forecasts first
	assert T.balanced(out) == out and T.balanced([fc(0, 'yes'), fc(1, 'no')]) == [fc(0, 'yes'), fc(1, 'no')]  # idempotent; balanced sets pass
	assert [t['id'] for t in T.balanced([fc(i, 'yes') for i in range(3)])] == []  # one-sided forecasts are no forecasting test


def test_calibration_steers_toward_the_minority_outcome_and_shows_rejected_forecasts():
	txt = T.calibration([fc(i, 'yes') for i in range(5)] + [fc(5, 'no'), fc(6, 'yes', verdict='no_shift')])
	assert 'OUTCOME BALANCE: 5 of' in txt and 'resolve NO' in txt and 'REJECTED FORECASTS' in txt and '[no_shift]' in txt
	assert 'OUTCOME BALANCE' not in T.calibration([fc(i, 'yes' if i % 2 else 'no') for i in range(6)])


def test_env_serves_the_balanced_set(tmp_path):
	from rl_env_use.env import Env

	write_jl(tmp_path / 'tasks.jsonl', [fc(i, 'yes') for i in range(6)] + [fc(6, 'no'), fc(7, 'no')])
	got = Counter(t['answer'] for t in Env(tmp_path).tasks())
	assert got == {'yes': 3, 'no': 2}


def test_balanced_ignores_pool_order():
	pool = [fc(i, 'yes', case=f'c{i}') for i in range(8)] + [fc(8, 'no'), fc(9, 'no')]
	assert {t['id'] for t in T.balanced(pool)} == {t['id'] for t in T.balanced(list(reversed(pool)))}
