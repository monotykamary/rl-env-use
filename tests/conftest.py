import json
from pathlib import Path

import pytest

GT = Path(__file__).parent / 'ground_truth'


def jl(path: Path) -> list[dict]:
	return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_jl(path: Path, rows):
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class FakeBrowser:
	"""What the scout tools read from a browser: the current page's url, title and text."""

	def __init__(self, text: str = '', url: str = 'https://example.org/', title: str = 'Page'):
		self.page, self.at, self.name, self.url_ = text, url, title, 'fake'

	async def url(self):
		return self.at

	async def title(self):
		return self.name

	async def text(self):
		return self.page


def tool_runner(tools):
	"""run(name, **args) -> NS(error, extracted_content), the way the agent sees a tool's result."""
	import asyncio
	from types import SimpleNamespace as NS

	from rl_env_use.agent import ToolError

	by = {t.name: t for t in tools}

	def run(name, **kw):
		t = by[name]
		try:
			return NS(error=None, extracted_content=asyncio.run(t.fn(**dict(t.params(**kw)))))
		except ToolError as e:
			return NS(error=str(e), extracted_content=None)

	return run


def mk_task(i, answer='137', answer_type='number', category='lookup', **kw):
	t = {
		'id': f't{i}',
		'prompt': f'Question {i} about benchmark X?',
		'start_url': 'https://arxiv.org/',
		'answer': answer,
		'answer_type': answer_type,
		'aliases': [],
		'fact_ids': [f'f{i}'],
		'derivation': '',
		'category': category,
		'difficulty': 'easy',
		'why_hard': 'buried',
		'checks': {
			'structural': [],
			'leakage': False,
			'closed_book': {'haiku': False},
			'live': True,
			'critic': {'unambiguous': True, 'gold_supported': True, 'realistic': True, 'issues': ''},
		},
	}
	t.update(kw)
	return t


def mk_row(tid, rep, model, correct, claimed=True):
	return {
		'prompt_id': tid,
		'rep': rep,
		'prompt': f'Question {tid}',
		'tags': ['lookup', 'easy'],
		'grade': {'correct': float(correct)},
		'status': 'ok',
		'model': model,
		'latency_s': 10.0,
		'tool_calls': 4,
		'steps': 5,
		'usage': {'input_tokens': 1000, 'output_tokens': 100},
		'cost_usd': 0.0015,
		'meta': {'answer': '137' if correct else '12', 'gold': '137', 'failure_class': None if correct else 'wrong_answer', 'agent_claimed_success': claimed},
	}


@pytest.fixture
def env_dir(tmp_path) -> Path:
	"""Synthetic env: 20 clean tasks + 3 filtered ones, a weak and a strong model run, traces and annotations."""
	env = tmp_path / 'env'
	tasks = [mk_task(i, category='lookup' if i % 2 else 'multi_hop') for i in range(20)]
	ok = mk_task(0)['checks']
	tasks.append(mk_task(20, checks={**ok, 'leakage': True}))
	tasks.append(mk_task(21, checks={**ok, 'closed_book': {'sonnet': True}}))
	tasks.append(mk_task(22, checks={**ok, 'live': False}))
	write_jl(env / 'tasks.jsonl', tasks)
	kb = {
		'topic': 'x',
		'facts': [{'id': f'f{i}', 'claim': 'c', 'value': '137', 'quote': 'from 137 websites', 'url': 'https://arxiv.org/abs/2306.06070'} for i in range(23)],
		'ideas': [],
		'sources': [],
		'missions': [],
	}
	(env / 'kb.json').write_text(json.dumps(kb))
	(env / 'spec.json').write_text(json.dumps({'topic': 'x'}))
	flow = env / 'hillclimb'
	# weak model: 30%, strong: 70%; t0 is inverted (weak right, strong wrong)
	weak = [mk_row(f't{i}', 0, 'anthropic/claude-haiku-4-5', i in (0, 1, 2, 3, 4, 5)) for i in range(20)]
	strong = [mk_row(f't{i}', 0, 'anthropic/claude-sonnet-4-6', i in range(1, 15)) for i in range(20)]
	write_jl(flow / 'baseline' / 'results.jsonl', weak)
	write_jl(flow / 'v1' / 'results.jsonl', strong)
	for r in weak:
		tr = [
			{'role': 'system', 'content': 'task'},
			{'role': 'assistant', 'content': 'step'},
			{'role': 'tool_call', 'name': 'navigate', 'content': '{"url": "https://arxiv.org"}'},
			{'role': 'tool_result', 'content': 'ok'},
			{'role': 'assistant', 'content': f'FINAL ANSWER: {r["meta"]["answer"]}'},
		]
		(flow / 'baseline' / 'traces').mkdir(parents=True, exist_ok=True)
		(flow / 'baseline' / 'traces' / f'{r["prompt_id"]}_rep0.json').write_text(json.dumps(tr))
	ann = [
		{
			'id': f'baseline/t{i}_rep0',
			'verdict': 'pass' if i < 6 else 'fail',
			'notes': '' if i < 6 else 'stopped on the search results page',
			'modes': [] if i < 6 else ['premature_stop'],
		}
		for i in range(12)
	]
	write_jl(env / 'annotations.jsonl', ann)
	return env


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
	"""No test may reach a paid service: no API key (main()'s .env loading never overrides a set variable), no local claude
	or codex CLI, no Modal. A test that needs a route fakes it and the client. Only `-m live` tests reach the real ones."""
	if request.node.get_closest_marker('live'):
		return
	import rl_env_use.cloud as cloud
	import rl_env_use.llm as L

	monkeypatch.setenv('ANTHROPIC_API_KEY', '')
	monkeypatch.setattr(L.shutil, 'which', lambda b: None)
	monkeypatch.setattr(cloud, 'configured', lambda: False)
	import rl_env_use.pipeline as P

	async def offline(url, *a, **k):  # look() fetches every mined page: never from a test
		return {'text': '', 'raw': '', 'blocked': True}

	monkeypatch.setattr(P, 'http_capture', offline)
	import rl_env_use.scout as S

	monkeypatch.setattr(S, 'http_capture', offline)
