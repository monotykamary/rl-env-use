"""Case tasks: browser agents mine (materials, outcome) cases; the policy reads the materials and answers with no browser."""

import asyncio
import json

from conftest import FakeBrowser, jl, mk_task, tool_runner

import rl_env_use.checks as C
import rl_env_use.rollout as R
import rl_env_use.scout as S
import rl_env_use.tasks as T
import rl_env_use.verify as V
from rl_env_use.__main__ import main, write_state
from rl_env_use.env import Env

PAGE = (
	'Header junk. ITEM 2. LIQUIDITY Cash fell to $4.1 million while the company burned $3.0 million per quarter. '
	+ 'Filler sentence about operations. ' * 12
	+ 'We may need to raise additional capital. Footer.'
)
CTX = 'ITEM 2. LIQUIDITY Cash fell to $4.1 million while the company burned $3.0 million per quarter.'


def case_task(i=0, **kw):
	t = mk_task(
		i,
		answer='substantial doubt',
		answer_type='contains',
		context=CTX,
		kind='predict',
		case='acme-2024',
		material_ids=['m0'],
		start_url='',
		prompt='Will the auditor raise going-concern doubt, and in what words?',
	)
	return t | kw


def test_slice_passage_is_verbatim_and_bounded():
	p = S.slice_passage(PAGE, 'ITEM 2. LIQUIDITY', 'raise additional capital.')
	assert p.startswith('ITEM 2.') and p.endswith('capital.') and p in PAGE
	assert S.slice_passage(PAGE, 'ITEM 7', 'capital.').startswith('!starts_with')
	assert S.slice_passage(PAGE, 'ITEM 2. LIQUIDITY', 'not on page').startswith('!ends_with')
	assert S.slice_passage(PAGE, 'ITEM 2. LIQUIDITY', 'per quarter.').startswith('!passage is')  # too short to be a material


def test_record_material_and_outcome_fact_share_a_case(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': []}
	run = tool_runner(S.make_tools(kb, 'cases', FakeBrowser(PAGE, 'https://www.sec.gov/Archives/acme-10q.htm')))
	assert run('record_material', case='acme-2024', title='Acme 10-Q liquidity', starts_with='ITEM 9', ends_with='capital.').error
	assert not run('record_material', case='acme-2024', title='Acme 10-Q liquidity', starts_with='ITEM 2. LIQUIDITY', ends_with='raise additional capital.').error
	assert not run('record_fact', claim='cash', value='$4.1 million', quote='Cash fell to $4.1 million while', case='acme-2024').error
	assert kb['materials'][0]['text'].startswith('ITEM 2.') and kb['facts'][0]['case'] == 'acme-2024'
	assert list(T.cases(kb)) == ['acme-2024']
	kb['materials'].append(kb['materials'][0] | {'id': 'm1', 'case': 'no-outcome'})
	assert list(T.cases(kb)) == ['acme-2024']  # a case without its outcome is not a case


def test_synthesize_cases_builds_context_from_chosen_materials(monkeypatch):
	kb = {
		'topic': 'credit',
		'skill': 'predict going-concern',
		'materials': [{'id': 'm0', 'case': 'acme', 'title': 'Acme 10-Q', 'text': CTX, 'url': 'https://sec.gov/a'}],
		'facts': [
			{
				'id': 'f0',
				'case': 'acme',
				'claim': 'opinion',
				'value': 'substantial doubt',
				'quote': 'raise substantial doubt about its ability to continue',
				'url': 'https://sec.gov/b',
			}
		],
	}
	seen = {}

	async def call(model, system, user, schema, **kw):
		seen['user'] = user
		d = T.CaseTaskDraft(
			prompt='Will the next annual report carry a going-concern warning, and what phrase will the auditor use?',
			kind='predict',
			material_ids=['m0', 'bogus'],
			answer='substantial doubt',
			answer_type='contains',
			fact_ids=['f0'],
			category='going_concern',
			difficulty='hard',
			why_hard='needs a burn-rate read',
		)
		return T.CaseDrafts(tasks=[d]), {}, 'm'

	monkeypatch.setattr(T, 'call', call)
	[t] = asyncio.run(T.synthesize_cases(kb))
	assert 'OUTCOME FACTS' in seen['user'] and t['context'] == '### Acme 10-Q\n' + CTX and t['material_ids'] == ['m0'] and t['case'] == 'acme'
	assert t['checks'] == {'structural': [], 'leakage': False} and t['evidence'][0]['url'] == 'https://sec.gov/b'


def test_leak_into_context_and_case_critic_gates():
	assert C.leaks(case_task(answer='burned 3.0 million per quarter'))  # a multi-word gold quoted in the materials
	assert not C.leaks(case_task())
	ok = {
		'structural': [],
		'leakage': False,
		'live': True,
		'closed_book': {'haiku': False},
		'critic': {'unambiguous': True, 'gold_supported': True, 'agrees': True, 'answer_in_context': False, 'well_posed': True},
	}
	assert T.keep(case_task(checks=ok))
	assert not T.keep(case_task(checks=ok | {'critic': ok['critic'] | {'answer_in_context': True}}))
	assert not T.keep(case_task(checks=ok | {'critic': ok['critic'] | {'well_posed': False}}))
	assert C.structural_issues(case_task(material_ids=[]), {'f0': {'quote': 'raise substantial doubt', 'url': 'u'}}) == ['case task without materials']


def test_closed_book_skips_coin_flips(monkeypatch):
	async def call(model, system, user, schema, **kw):
		return schema(answer='yes'), {}, 'm'

	monkeypatch.setattr(V, 'call', call)
	flip, open_q = case_task(0, answer='yes', answer_type='exact'), case_task(1)
	asyncio.run(V.closed_book([flip, open_q], ['haiku']))
	assert flip['checks']['closed_book'] == 'skipped' and open_q['checks']['closed_book'] == {'haiku': False}


def test_case_policy_sees_context_never_gold(tmp_path, monkeypatch):
	t = case_task()
	assert set(R.public(t)) == {'id', 'prompt', 'context'}
	got = {}

	async def call(model, system, user, schema, **kw):
		got['user'] = user
		return schema(reasoning='burn exceeds cash', answer='Substantial doubt'), {'input_tokens': 100, 'output_tokens': 20}, 'anthropic/claude-sonnet-4-6'

	monkeypatch.setattr(R, 'call', call)
	vdir = asyncio.run(R.rollout([t], tmp_path / 'hc', model='sonnet46'))
	[row] = jl(vdir / 'results.jsonl')
	trace = json.loads((vdir / 'traces' / 't0_rep0.json').read_text())
	assert CTX in got['user'] and 'substantial doubt' not in got['user'].lower()
	assert row['grade'] == {'correct': 1.0} and row['tags'] == ['lookup', 'predict', 'easy'] and row['cost_usd'] > 0 and row['model'] == 'anthropic/claude-sonnet-4-6'
	assert [x['role'] for x in trace] == ['system', 'user', 'assistant'] and trace[-1]['thinking'] == 'burn exceeds cash'


def test_skill_env_spec_metrics_and_rollout(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	main(['new', 'Going Concern', '--topic', 'credit analysis', '--skill', 'given a 10-Q, predict the auditor opinion', '--sources', 'https://www.sec.gov/'])
	env = tmp_path / 'envs/going-concern'
	s = json.loads((env / 'spec.json').read_text())
	assert s['skill'].startswith('given a 10-Q') and s['tasks_per_case'] == 2 and 'n_tasks' not in s
	write_state(env)
	assert [m['id'] for m in json.loads((env / 'hillclimb/_state.json').read_text())['metrics']] == ['correct']
	ok = {'structural': [], 'leakage': False, 'live': True, 'closed_book': {'haiku': False}, 'critic': {'unambiguous': True, 'gold_supported': True, 'agrees': True}}
	(env / 'tasks.jsonl').write_text(json.dumps(case_task(checks=ok)) + '\n')

	async def call(model, system, user, schema, **kw):
		return schema(reasoning='r', answer='no doubt at all'), {}, 'm'

	monkeypatch.setattr(R, 'call', call)  # Env.rollout answers through rollout.solve_case
	out = asyncio.run(Env(env).rollout('t0'))
	assert out['reward'] == 0.0 and out['answer'] == 'no doubt at all'
	assert Env(env).export_jsonl(env / 'd.jsonl') == 1 and 'context' in jl(env / 'd.jsonl')[0] and not (env / 'agent_tasks').exists()


def test_vanilla_case_materials_are_cut_by_us_from_the_live_page(monkeypatch):
	import rl_env_use.vanilla as V

	class Sess:
		async def start(self):
			pass

		async def kill(self):
			pass

	async def capture(session, url):
		return {'text': PAGE}

	monkeypatch.setattr(V, 'browser', Sess)
	monkeypatch.setattr(V, 'capture', capture)
	kb = {'facts': []}
	mats = [
		{'case': 'acme', 'title': 'Acme 10-Q', 'url': 'https://sec.gov/a', 'starts_with': 'ITEM 2. LIQUIDITY', 'ends_with': 'raise additional capital.'},
		{'case': 'acme', 'title': 'invented', 'url': 'https://sec.gov/a', 'starts_with': 'ITEM 9', 'ends_with': 'capital.'},
	]
	facts = [{'case': 'acme', 'claim': 'cash', 'value': '$4.1 million', 'quote': 'Cash fell to $4.1 million while', 'url': 'https://sec.gov/a'}]
	assert asyncio.run(V.verify_facts(facts, kb, 'cases', 'claude-code', mats)) == 1
	assert [m['title'] for m in kb['materials']] == ['Acme 10-Q'] and kb['materials'][0]['text'] in PAGE and list(T.cases(kb)) == ['acme']


def test_scout_finishes_started_cases_before_new_ones():
	kb = {
		'facts': [{'id': 'f0', 'case': 'done', 'claim': 'c', 'value': '1', 'quote': 'q'}],
		'materials': [
			{'id': 'm0', 'case': 'done', 'title': 'A', 'url': 'https://sec.gov/a', 'text': 'x'},
			{'id': 'm1', 'case': 'open', 'title': 'B 10-Q', 'url': 'https://sec.gov/b', 'text': 'x'},
			{'id': 'm2', 'case': 'open', 'title': 'B notes', 'url': 'https://sec.gov/b2', 'text': 'x'},
		],
	}
	[m] = S.outcome_missions(kb, 'predict')
	assert m.angle == 'outcome' and m.start_url == 'https://sec.gov/b' and 'case="open"' in m.instructions
	assert 'Do not collect new cases' in S.case_task('credit', 'predict', m) and 'Aim for' not in S.case_task('credit', 'predict', m)


def test_forecasts_get_a_band_derivations_stay_exact():
	f = T.finish(
		{
			'prompt': 'Forecast year-end cash ($)?',
			'kind': 'predict',
			'answer': '4674572',
			'answer_type': 'number',
			'fact_ids': [],
			'derivation': '',
			'context': CTX,
			'material_ids': ['m0'],
			'case': 'c',
		},
		{},
	)
	assert f['rel_tol'] == 0.10 and C.grade_answer(f, '4.4 million') and not C.grade_answer(f, '3.9 million')
	d = T.finish(
		{
			'prompt': 'Current ratio?',
			'kind': 'derive',
			'answer': '2.73',
			'answer_type': 'number',
			'fact_ids': [],
			'derivation': '101766/37279',
			'context': CTX,
			'material_ids': ['m0'],
			'case': 'c',
		},
		{},
	)
	assert d['abs_tol'] == 0.005 and not C.grade_answer(d, '2.8')
	assert 'tolerance too loose (rel_tol > 15%)' in C.structural_issues(f | {'rel_tol': 0.3}, {}) and C.structural_issues(d | {'rel_tol': 0.1}, {})


def test_materials_capped_until_the_outcome_is_recorded(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': [], 'materials': [{'id': f'm{i}', 'case': 'acme', 'title': 't', 'text': f'other {i}', 'url': 'u'} for i in range(3)]}
	b = FakeBrowser(PAGE, 'https://www.sec.gov/a.htm')
	run = tool_runner(S.make_tools(kb, 'outcome', b))
	other = tool_runner(S.make_tools(kb, 'cases', b))  # a new-case mission may not reuse another case's key
	assert 'already taken' in other('record_material', case='acme', title='x', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').error
	assert 'record its outcome' in run('record_material', case='acme', title='x', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').error
	assert not run('record_fact', claim='cash', value='$4.1 million', quote='Cash fell to $4.1 million while', case='acme').error
	assert not run('record_material', case='acme', title='x', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').error


def test_record_fact_without_value_takes_the_single_number(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': []}
	page = 'Overall Study Started 28 participants; Completed 26 participants. Mean change in body weight Tirzepatide -20.9 % and placebo -3.1 %'
	run = tool_runner(S.make_tools(kb, 'cases', FakeBrowser(page, 'https://clinicaltrials.gov/study/NCT1?tab=results')))
	assert not run('record_fact', claim='completed', quote='Completed 26 participants. Mean change', case='nct1').error
	assert 'Give `value`' in run('record_fact', claim='weight change', quote='Tirzepatide -20.9 % and placebo -3.1 %', case='nct1').error
	assert kb['facts'][0]['value'] == '26'


def test_outcome_must_use_a_known_case_key(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': [], 'materials': [{'id': 'm0', 'case': 'NCT1_protocol', 'title': 't', 'text': 'x', 'url': 'u'}]}
	run = tool_runner(S.make_tools(kb, 'cases', FakeBrowser('Enrollment (Actual) 28 participants', 'https://clinicaltrials.gov/study/NCT1')))
	assert 'NCT1_protocol' in run('record_fact', claim='enrollment', quote='Enrollment (Actual) 28', value='28', case='NCT1_results').error
	assert not run('record_fact', claim='enrollment', quote='Enrollment (Actual) 28', value='28', case='NCT1_protocol').error
	kb['facts'].append({'id': 'f1', 'case': 'no-materials', 'claim': 'c', 'value': '1', 'quote': 'q', 'url': 'https://x.org/y'})
	[m] = S.enrich_missions(kb)  # complete cases only, starting from where the outcome was found
	assert m.angle == 'enrich' and m.start_url == 'https://clinicaltrials.gov/study/NCT1' and 'case="NCT1_protocol"' in m.instructions


def test_derived_gold_is_computed_by_code():
	from rl_env_use.checks import evaluate, expression_issue

	t = {'kind': 'derive', 'answer_type': 'number', 'context': 'Arm A 10,875; arm B 6,939.', 'prompt': 'Share of arm A, in percent?', 'expression': '10875 / (10875 + 6939) * 100'}
	assert expression_issue(t | {'answer': '61.0'}) is None and 'does not match' in expression_issue(t | {'answer': '61.1'})
	assert expression_issue(t | {'answer': '61', 'expression': ''}) and evaluate('__import__("os")') is None and evaluate('48,809,930 / 3,246,371') > 15.035
	assert expression_issue(t | {'kind': 'predict', 'answer': '70'}) is None


def test_crashed_outcome_mission_is_retried_but_a_finished_one_is_not():
	kb = {
		'facts': [],
		'materials': [{'id': 'm0', 'case': 'acme', 'title': 't', 'text': 'x', 'url': 'https://a/0'}],
		'missions': [{'angle': 'outcome', 'instructions': 'Case `acme` already has...', 'error': 'HTTP 503'}],
	}
	assert [m.start_url for m in S.outcome_missions(kb, 's')] == ['https://a/0']
	kb['missions'].append({'angle': 'outcome', 'instructions': 'Case `acme` already has...', 'done': True})
	assert S.outcome_missions(kb, 's') == []


def test_fact_sign_must_match_the_page():
	assert 'minus' in S.fact_issue('-4.5', 'Net income was 4.5 million in 2024')
	assert 'with its sign' in S.fact_issue('4.5', 'Net loss was -4.5 million in 2024')
	assert S.fact_issue('-4.5', 'Net loss was (4.5) million in 2024') is None


def test_record_outcome_saves_a_verbatim_results_passage(monkeypatch):
	kb = {'facts': [], 'ideas': [], 'sources': [], 'materials': [{'id': 'm0', 'case': 'acme', 'title': 't', 'text': 'x', 'url': 'u'}]}
	tools = tool_runner(S.make_tools(kb, 'cases', FakeBrowser(PAGE, 'https://www.sec.gov/a.htm')))
	run = lambda **kw: tools('record_outcome', **kw)  # noqa: E731
	assert 'Unknown case' in run(case='beta', title='r', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').error
	assert not run(case='acme', title='results', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').error
	f = kb['facts'][0]
	assert f['case'] == 'acme' and f['passage'] and f['quote'].startswith('ITEM 2. LIQUIDITY') and f['quote'].endswith('capital.')
	assert 'Already saved' in run(case='acme', title='results', starts_with='ITEM 2. LIQUIDITY', ends_with='capital.').extracted_content and len(kb['facts']) == 1


def test_a_stuck_mission_times_out_and_the_rest_continue(monkeypatch, tmp_path):
	async def run_mission(m, *a, **k):
		if m.start_url.endswith('stuck'):
			await asyncio.sleep(60)
		return {'angle': m.angle, 'start_url': m.start_url}

	async def plan(*a, **k):
		return [S.Mission(angle='cases', instructions='i', start_url=u) for u in ('https://a/stuck', 'https://a/ok')]

	monkeypatch.setattr(S, 'run_mission', run_mission)
	monkeypatch.setattr(S, 'plan', plan)
	monkeypatch.setattr(S, 'MISSION_TIMEOUT', 0.1)
	kb = asyncio.run(S.scout('t', tmp_path, 2, skill=None))
	got = {m['start_url']: m.get('error', '') for m in kb['missions']}
	assert got['https://a/ok'] == '' and got['https://a/stuck'].startswith('TimeoutError')
