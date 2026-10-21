import argparse
import json

import pytest
from conftest import jl, write_jl

from rl_env_use.__main__ import main
from rl_env_use.env import Env


def test_env_filters_and_hides_gold(env_dir):
	env = Env(env_dir)
	assert len(env) == 20 and 't20' not in env.ids
	obs = env.reset('t3')
	assert 'answer' not in obs and obs['start_url'].startswith('https://')
	assert env.reward('t3', 'It is 137.') == 1.0 and env.reward('t3', None) == 0.0
	assert len(Env(env_dir, allow_memorizable=True)) == 21


def test_env_applies_human_review(env_dir):
	write_jl(env_dir / 'reviews.jsonl', [{'id': 't3', 'verdict': 'accept', 'gold_fix': '138'}, {'id': 't4', 'verdict': 'reject'}])
	env = Env(env_dir)
	assert env.reward('t3', '138') == 1.0 and env._tasks['t3']['gold_source'] == 'human-corrected'
	assert 't4' not in env.ids


def test_cli_split_audit_export(env_dir, capsys):
	main(['split', str(env_dir), '--seed', '3'])
	s = json.loads((env_dir / 'hillclimb' / '_state.json').read_text())
	assert len(s['train_ids']) + len(s['test_ids']) == 20 and [m['id'] for m in s['metrics']][:2] == ['supported', 'correct']
	assert len(Env(env_dir, split='test')) == len(s['test_ids'])
	main(['report', str(env_dir), '--no-judges'])
	out = capsys.readouterr().out
	assert '(filtered out)' in out and 'grader self-test clean' in out and (env_dir / 'audit.json').exists()  # gradient logic: test_audit.py
	main(['export', str(env_dir)])
	rows = jl(env_dir / 'dataset.jsonl')
	assert len(rows) == 20 and 'checks' not in rows[0] and rows[0]['answer'] == '137'


def test_split_requires_state(env_dir):
	import pytest

	with pytest.raises(ValueError):
		Env(env_dir, split='test')


def test_labels_filter_by_annotator(env_dir):
	from rl_env_use.__main__ import labels

	write_jl(
		env_dir / 'annotations.jsonl',
		[{'id': 'a', 'verdict': 'fail'}, {'id': 'b', 'verdict': 'pass', 'annotator': 'sonnet-subagent'}, {'id': 'a', 'verdict': 'pass', 'annotator': 'human'}],
	)
	assert [(r['id'], r['verdict']) for r in labels(env_dir, 'human')] == [('a', 'pass')]
	assert [r['id'] for r in labels(env_dir, 'sonnet-subagent')] == ['b'] and len(labels(env_dir, 'any')) == 2


def test_regrade_flips_only_changed(env_dir, capsys):
	rows = jl(env_dir / 'hillclimb' / 'baseline' / 'results.jsonl')
	rows[0]['meta']['answer'], rows[0]['grade']['correct'] = '137', 0.0  # stale wrong grade
	write_jl(env_dir / 'hillclimb' / 'baseline' / 'results.jsonl', rows)
	main(['report', str(env_dir), '--no-judges'])
	assert jl(env_dir / 'hillclimb' / 'baseline' / 'results.jsonl')[0]['grade']['correct'] == 1.0


def test_regrade_restores_rows_of_readmitted_tasks(env_dir):
	write_jl(env_dir / 'reviews.jsonl', [{'id': 't3', 'verdict': 'reject'}])
	main(['report', str(env_dir), '--no-judges'])
	assert all(r['prompt_id'] != 't3' for r in jl(env_dir / 'hillclimb' / 'baseline' / 'results.jsonl'))
	write_jl(env_dir / 'reviews.jsonl', [{'id': 't3', 'verdict': 'accept'}])
	main(['report', str(env_dir), '--no-judges'])
	assert any(r['prompt_id'] == 't3' for r in jl(env_dir / 'hillclimb' / 'baseline' / 'results.jsonl'))
	assert jl(env_dir / 'hillclimb' / 'baseline' / 'dropped.jsonl') == []


def test_new_topic_spec_with_sources(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	main(
		['new', 'Rare Variants', '--topic', 'rare variants', '--sources', 'https://gnomad.broadinstitute.org/', 'https://www.uniprot.org/', '--proxy', 'http://proxy.example:3128']
	)
	s = json.loads((tmp_path / 'envs/rare-variants/spec.json').read_text())
	assert s['sources'][1] == 'https://www.uniprot.org/' and s['proxy'] == 'http://proxy.example:3128' and s['missions'] == 4
	with pytest.raises(SystemExit):
		main(['new', 'Rare Variants', '--topic', 'again'])
	with pytest.raises(SystemExit):
		main(['new', 'nothing'])


def test_rejecting_a_fact_drops_tasks_that_cite_it(env_dir):
	write_jl(env_dir / 'data_reviews.jsonl', [{'id': 'f3', 'verdict': 'reject'}])
	ids = Env(env_dir).ids
	assert 't3' not in ids and 't4' in ids


def test_run_applies_the_env_proxy(monkeypatch):
	import os

	from rl_env_use.browse import Browser
	from rl_env_use.pipeline import use_browser_spec

	monkeypatch.setenv('RLENV_PROXY', 'http://old:1')  # restored after the test
	use_browser_spec({'proxy': 'http://proxy.example:3128'})
	assert os.environ['RLENV_PROXY'] == 'http://proxy.example:3128' and Browser().proxy == 'http://proxy.example:3128'


def test_run_dry_run_estimates_without_spending(env_dir, capsys, monkeypatch):
	monkeypatch.chdir(env_dir.parent)
	main(['run', str(env_dir), '--model', 'haiku', '--dry-run'])
	main(['run', str(env_dir), '--model', 'qwen-35b', '--dry-run'])
	out = capsys.readouterr().out
	assert '20 episodes on haiku: ~$0.03, ~1 min at concurrency 4 (measured on 20 earlier episodes)' in out
	assert '20 episodes on qwen-35b: not measured on this env yet' in out and not (env_dir / 'hillclimb' / 'baseline' / 'errors.jsonl').exists()


def test_every_flow_is_regraded_and_audited(env_dir):
	import shutil

	from rl_env_use.__main__ import flows, report

	shutil.copytree(env_dir / 'hillclimb' / 'v1', env_dir / 'models' / 'v1')
	report(env_dir, judges=False)
	assert [f.name for f in flows(env_dir)] == ['hillclimb', 'models'] and (env_dir / 'models' / '_state.json').exists()
	msgs = [f['message'] for f in json.loads((env_dir / 'audit.json').read_text())]
	assert any(m.startswith('models/v1 (') for m in msgs) and any(m.startswith('v1 (') for m in msgs)


def test_regrade_keeps_rows_of_verified_tasks_the_panel_found_easy(tmp_path):
	import json

	from conftest import jl, mk_row, mk_task, write_jl

	from rl_env_use.__main__ import regrade

	env = tmp_path / 'e'
	ok = mk_task(0)['checks']
	write_jl(env / 'tasks.jsonl', [mk_task(0, checks=ok | {'panel': {'verdict': 'saturated'}}), mk_task(1, checks=ok | {'panel': {'verdict': 'multi_value'}})])
	(env / 'spec.json').write_text(json.dumps({'topic': 'x'}))
	write_jl(env / 'pool' / 'baseline' / 'results.jsonl', [mk_row('t0', 0, 'm', True), mk_row('t1', 0, 'm', True)])
	regrade(env, set())
	assert [r['prompt_id'] for r in jl(env / 'pool' / 'baseline' / 'results.jsonl')] == ['t0'] and [r['prompt_id'] for r in jl(env / 'pool' / 'baseline' / 'dropped.jsonl')] == [
		't1'
	]


def test_append_after_a_cut_line_starts_a_new_line(tmp_path):
	import asyncio

	from rl_env_use.runner import run_eval
	from rl_env_use.store import load_jsonl

	res = tmp_path / 'baseline' / 'results.jsonl'
	res.parent.mkdir()
	res.write_text('{"prompt_id": "a", "rep": 0, "status": "ok"}\n{"prompt_id": "b", "re')

	async def one(c, rep, vdir):
		return [], {'prompt_id': c['id'], 'rep': rep, 'status': 'ok'}

	asyncio.run(run_eval([{'id': 'a'}, {'id': 'b'}], one, tmp_path))
	assert sorted(r['prompt_id'] for r in load_jsonl(res)) == ['a', 'b']


def test_comparison_flow_gives_each_model_its_own_named_slot(tmp_path):
	from rl_env_use.__main__ import model_slot

	flow = tmp_path / 'models'
	write_jl(flow / 'baseline' / 'results.jsonl', [{'model': 'qwen/qwen3.6-35b-a3b'}])  # an older run without change.md
	assert model_slot(flow, 'qwen-35b') == 'baseline'
	assert model_slot(flow, 'opus55') == 'v1' and (flow / 'v1' / 'change.md').read_text() == 'opus55\n'
	assert model_slot(flow, 'gemini') == 'v2' and model_slot(flow, 'opus55') == 'v1'


def test_blind_run_compares_against_the_same_models_slot(tmp_path, monkeypatch):
	import rl_env_use.__main__ as M
	import rl_env_use.pipeline as P

	env = tmp_path / 'e'
	(env / 'models' / 'baseline').mkdir(parents=True)
	(env / 'models' / 'baseline' / 'change.md').write_text('haiku\n')
	(env / 'models' / 'v1').mkdir()
	(env / 'models' / 'v1' / 'change.md').write_text('opus55\n')
	(env / 'spec.json').write_text(json.dumps({'skill': 's'}))
	seen = {}

	async def fake_run(env_, spec, a):
		seen['variant'] = a.variant
		return env_ / 'ablation' / a.variant

	monkeypatch.setattr(P, 'run', fake_run)
	M.cmd_run(argparse.Namespace(env=str(env), model='opus55', variant=None, blind=True, flow=None, compare_flow='models', dry_run=False))  # main() would os._exit
	assert seen['variant'] == 'v1'
