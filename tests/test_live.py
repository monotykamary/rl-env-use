"""Ground-truth evals of the framework's own LLM/browser components. Paid + network: `pytest -m live`."""

import asyncio
import copy

import pytest
from conftest import GT, jl

from rl_env_use.audit import perturb
from rl_env_use.llm import load_dotenv
from rl_env_use.verify import check_live, critique

pytestmark = pytest.mark.live
load_dotenv()


@pytest.fixture(autouse=True)
def stop_modal():
	"""Whatever a test leaves running on Modal (a model server, a browser a failure skipped) is stopped right after it."""
	from rl_env_use import cloud

	yield
	asyncio.run(cloud.shutdown())


def gt_env():
	tasks, facts = jl(GT / 'web_tasks.jsonl'), {}
	for t in tasks:
		t['fact_ids'], t['checks'] = [], {'structural': [], 'leakage': False}
		for e in t['evidence']:
			fid = f'{t["id"]}_{len(t["fact_ids"])}'
			facts[fid] = {'id': fid, **e}
			t['fact_ids'].append(fid)
	return tasks, facts


def test_ground_truth_evidence_still_live():
	"""Every human-verified quote must be found by the live checker (catches staleness AND checker bugs)."""
	tasks, facts = gt_env()
	asyncio.run(check_live(tasks, facts))
	assert [t['id'] for t in tasks if not t['checks']['live']] == []


AMBIGUOUS = [
	('gt_webarena_v1', 'When was the WebArena paper published?'),  # v1 vs v4 vs venue
	('gt_mind2web_sites', 'How many websites does Mind2Web have?'),  # abstract vs dataset release vs test splits
	('gt_osworld_tasks', 'How many tasks does OSWorld have?'),  # benchmark vs verified subset vs later versions
]


def test_critic_against_ground_truth():
	"""Critic must accept verified tasks, reject corrupted golds, and flag under-specified prompts."""
	tasks, facts = gt_env()
	clean = copy.deepcopy(tasks)
	wrong = [t for t in copy.deepcopy(tasks) if perturb(t)]
	for t in wrong:
		t['answer'] = perturb(t)
	amb = []
	for tid, prompt in AMBIGUOUS:
		t = copy.deepcopy(next(x for x in tasks if x['id'] == tid))
		t['prompt'] = prompt
		amb.append(t)
	asyncio.run(critique(clean + wrong + amb, facts))
	ok = lambda t: t['checks']['critic']['unambiguous'] and t['checks']['critic']['gold_supported']  # noqa: E731
	accept = sum(ok(t) for t in clean) / len(clean)
	catch_wrong = sum(not t['checks']['critic']['gold_supported'] for t in wrong) / len(wrong)
	catch_amb = sum(not t['checks']['critic']['unambiguous'] for t in amb) / len(amb)
	print(f'critic: accepts {accept:.0%} of verified tasks, rejects {catch_wrong:.0%} of corrupted golds, flags {catch_amb:.0%} of ambiguous prompts')
	assert accept >= 0.85 and catch_wrong >= 0.9 and catch_amb >= 2 / 3


def test_policy_smoke(tmp_path):
	from rl_env_use.rollout import rollout

	tasks = [t for t in jl(GT / 'web_tasks.jsonl') if t['id'] in ('gt_mind2web_sites', 'gt_browsecomp_q', 'gt_webarena_v1')]
	v = asyncio.run(rollout(tasks, tmp_path, model='sonnet46', max_steps=12, concurrency=3))
	rows = jl(v / 'results.jsonl')
	assert len(rows) == 3 and all(r['usage']['input_tokens'] > 0 and r['latency_s'] > 0 for r in rows)
	assert sum(r['grade']['correct'] for r in rows) >= 2, [r['meta'] for r in rows]
