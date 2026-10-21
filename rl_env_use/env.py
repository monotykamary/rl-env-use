"""The product: an RL environment with verifiable rewards, plus exporters. Case tasks hand the model a case's
materials (`context`); browse tasks hand it a start URL and a browser."""

import json
from pathlib import Path

from .graders import grade_answer
from .rollout import public, solve_case
from .tasks import balanced, keep, load_tasks


class Env:
	"""One episode = one task. Observation = public task fields (prompt + context, or prompt + start URL); reward = ground-truth check
	(+ optional calibrated failure-mode judges). Gold answers never appear in observations."""

	def __init__(self, env_dir, split: str | None = None, allow_memorizable: bool = False, verified: bool = False):
		self.dir = Path(env_dir)
		tasks = balanced([t for t in load_tasks(self.dir) if keep(t, allow_memorizable, verified)])
		if split:
			state = self.dir / 'hillclimb' / '_state.json'
			ids = json.loads(state.read_text()).get(f'{split}_ids') if state.exists() else None
			if ids is None:
				raise ValueError(f'no {split} split recorded - run `split` first')
			tasks = [t for t in tasks if t['id'] in set(ids)]
		self._tasks = {t['id']: t for t in tasks}

	def __len__(self):
		return len(self._tasks)

	@property
	def ids(self) -> list[str]:
		return sorted(self._tasks)

	def tasks(self) -> list[dict]:
		return [self._tasks[i] for i in self.ids]

	def reset(self, task_id: str) -> dict:
		return public(self._tasks[task_id])

	def reward(self, task_id: str, answer) -> float:
		return float(grade_answer(self._tasks[task_id], answer))

	async def rollout(self, task_id: str, model: str = 'sonnet46', max_steps: int = 25, **agent_kwargs) -> dict:
		obs = self.reset(task_id)
		if 'context' in obs:
			_, _, answer, reasoning, _, _ = await solve_case(obs, model)
			return {'task_id': task_id, 'answer': answer, 'reward': self.reward(task_id, answer), 'reasoning': reasoning}
		from .rollout import browse_episode

		h, _ = await browse_episode(obs, model, max_steps, agent_kwargs.get('extend_system'))
		answer = h.output.answer if h.output else None
		return {'task_id': task_id, 'answer': answer, 'reward': self.reward(task_id, answer), 'trace': h.trace}

	def export_jsonl(self, path) -> int:
		"""Flat dataset for external RL trainers: prompt + context (or start_url) + verifiable answer spec."""
		keys = (
			'id',
			'prompt',
			'context',
			'kind',
			'case',
			'start_url',
			'answer',
			'answer_type',
			'aliases',
			'rel_tol',
			'abs_tol',
			'min_recall',
			'category',
			'difficulty',
			'why_hard',
			'source_urls',
		)
		rows = [{k: t[k] for k in keys if k in t} for t in self.tasks()]
		Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))
		return len(rows)
