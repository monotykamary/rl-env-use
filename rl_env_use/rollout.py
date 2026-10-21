"""Stage 3: run the policy over tasks and write results in the claude-api hillclimb contract. Case tasks: the model
reads the materials and answers (one structured call, no tools). Browse tasks: a browser agent on the live web (agent.py).

<flow>/<variant>/results.jsonl     one row per (case, rep), appended as each completes (resume-safe)
<flow>/<variant>/traces/<id>_rep<k>.json
<flow>/<variant>/errors.jsonl      infra failures (timeouts, crashes) - never scored as model failures
"""

import time
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from .agent import run_agent
from .browse import browser
from .graders import grade_answer, grounded, max_repeats
from .llm import add_usage, call, cost, now
from .runner import run_eval

LOOP = 5  # identical tool calls at which a trajectory counts as looping

POLICY_SUFFIX = '\n\nReturn just the answer (value, name, date or comma-separated list) in `answer` - no explanation - and the page you found it on in `source_url`.'


CASE_SYS = 'You are a domain expert. Read the materials and answer the question. Reason it through, then commit to one answer.'
CASE_SUFFIX = '\n\nPut your work in `reasoning` and just the final answer (value, name, date or comma-separated list) in `answer`, no explanation.'
ANSWER_SUFFIX = '\n\nThink it through, then give just the final answer (value, name, date or comma-separated list) in `answer`, no explanation.'


class Reasoned(BaseModel):
	reasoning: str = Field(description='Your work: what in the materials you used and how you reached the answer.')
	answer: str = Field(description='Just the final answer, no explanation.')


class Answer(BaseModel):
	answer: str = Field(description='Just the final answer, no explanation.')
	source_url: str = ''


def trace_grade(t: dict, correct: bool, trace: list[dict]) -> dict:
	"""Reward + process metrics. `supported` = right answer AND the agent opened a source page: the reward to train on,
	since correctness alone rewards copying search snippets (found in error analysis)."""
	g = grounded(t.get('evidence') or [{'url': u} for u in t.get('source_urls', [])], trace)
	return {'correct': float(correct), 'supported': float(correct and g), 'grounded': float(g), 'no_loop': float(max_repeats(trace) < LOOP)}


def public(t: dict) -> dict:
	"""The only task fields a policy ever sees. Gold, evidence and checks stay out of reach."""
	if t.get('context'):
		return {'id': t['id'], 'prompt': t['prompt'], 'context': t['context']}
	return {'id': t['id'], 'prompt': t['prompt'], 'start_url': t['start_url'], 'allowed_domains': t.get('allowed_domains')}


def case_messages(pub: dict, extend_system: str | None = None, answer_only: bool = False) -> tuple[str, str]:
	return CASE_SYS + (f'\n\n{extend_system}' if extend_system else ''), f'MATERIALS:\n{pub["context"]}\n\nQUESTION: {pub["prompt"]}{ANSWER_SUFFIX if answer_only else CASE_SUFFIX}'


async def solve_case(pub: dict, model: str, extend_system: str | None = None):
	"""One policy answer on a case task: (system, user, answer, written reasoning, usage, model id). Frontier closed models
	answer only (their reasoning stays internal); the rest write their work, which the hillclimb's error analysis reads."""
	from .llm import ANSWER_ONLY

	quiet = model in ANSWER_ONLY
	system, user = case_messages(pub, extend_system, quiet)
	r, usage, mid = await call(model, system, user, Answer if quiet else Reasoned, temperature=None)
	return system, user, r.answer, getattr(r, 'reasoning', ''), usage, mid


async def run_case_task(t: dict, rep: int, model: str, judges=None, extend_system: str | None = None) -> tuple[list, dict]:
	t0 = time.monotonic()
	system, user, answer, reasoning, usage, mid = await solve_case(public(t), model, extend_system)
	r = Reasoned(reasoning=reasoning, answer=answer)
	correct = grade_answer(t, r.answer)
	trace = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': f'FINAL ANSWER: {r.answer}', 'thinking': r.reasoning}]
	grade, expl, judge_usage, judge_model = {'correct': float(correct)}, {}, {}, None
	for j in judges or []:
		v = await j.run(t, trace, r.answer)
		grade[j.id], expl[j.id], judge_model = v['score'], v['reasoning'], v['model']
		judge_usage = add_usage(judge_usage, v['usage'])
	return trace, {
		'prompt_id': t['id'],
		'rep': rep,
		'prompt': t['prompt'],
		'tags': [t.get('category', 'task'), t.get('kind', '?'), t.get('difficulty', '?')],
		'grade': grade,
		**({'explanation': expl} if expl else {}),
		'status': 'ok',
		'stop_reason': 'done',
		'model': mid,
		'latency_s': round(time.monotonic() - t0, 2),
		'tool_calls': 0,
		'steps': 1,
		'usage': usage,
		'cost_usd': round(cost(mid, usage) or 0, 5),
		**({'judge_model': judge_model, 'judge_usage': judge_usage} if judge_usage else {}),
		'meta': {'answer': r.answer, 'gold': t['answer'], 'failure_class': None if correct else 'wrong_answer', 'case': t.get('case'), 'at': now()},
	}


async def browse_episode(pub: dict, model: str, max_steps: int, extend_system: str | None = None):
	"""One browse-task episode on a fresh browser, always stopped afterwards: (run, the task text the agent got)."""
	task = pub['prompt'] + POLICY_SUFFIX
	b = browser(pub.get('allowed_domains'))
	try:
		await b.start()
		h = await run_agent(
			model,
			task,
			b,
			Answer,
			max_steps=max_steps,
			system=extend_system,
			start_url=pub['start_url'],
			done_description='Give your final answer. Call exactly once, as the last action.',
		)
	finally:  # a cancelled episode (wall-clock ceiling) must still stop its browser
		await b.kill()
	return h, task


async def run_case(t: dict, rep: int, model: str, vdir: Path, max_steps: int, headless: bool = True, judges=None, extend_system: str | None = None) -> tuple[list, dict]:
	t0 = time.monotonic()
	h, task = await browse_episode(public(t), model, max_steps, extend_system)
	latency = time.monotonic() - t0
	if not h.done and h.steps < max_steps and h.errors:  # stopped on tool crashes before its budget: infra, not a miss
		raise RuntimeError(f'agent stopped on errors: {h.errors[-1][:300]}')
	out = h.output
	answer = out.answer if out else None
	correct = grade_answer(t, answer)
	trace = h.trace
	trace.append({'role': 'assistant', 'content': f'FINAL ANSWER: {answer}' + (f'\nsource: {out.source_url}' if out and out.source_url else '')})
	usage = h.usage
	grade, expl, judge_usage, judge_model = trace_grade(t, correct, trace), {}, {}, None
	for j in judges or []:
		v = await j.run(t, trace, answer)
		grade[j.id], expl[j.id], judge_model = v['score'], v['reasoning'], v['model']
		judge_usage = add_usage(judge_usage, v['usage'])
	fc = None if correct else ('no_answer' if answer is None else 'wrong_answer')
	domains = list(dict.fromkeys(urlparse(u).netloc for u in h.urls if u))
	return trace, {
		'prompt_id': t['id'],
		'rep': rep,
		'prompt': t['prompt'],
		'tags': [t.get('category', 'task'), t.get('difficulty', '?')],
		'grade': grade,
		**({'explanation': expl} if expl else {}),
		'status': 'ok',
		'stop_reason': h.stop_reason,
		'model': h.model,
		'latency_s': round(latency, 2),
		'tool_calls': h.steps,
		'steps': h.steps,
		'usage': usage,
		'cost_usd': round(cost(h.model, usage) or 0, 5),
		**({'judge_model': judge_model, 'judge_usage': judge_usage} if judge_usage else {}),
		'meta': {
			'answer': answer,
			'gold': t['answer'],
			'failure_class': fc,
			'agent_claimed_success': h.done,
			'domains': domains,
			'errors': h.errors[:5],
			'at': now(),
		},
	}


async def rollout(
	tasks: list[dict],
	flow: Path,
	variant: str = 'baseline',
	model: str = 'sonnet46',
	reps: int = 1,
	max_steps: int = 25,
	timeout_s: float = 600,
	concurrency: int = 4,
	headless: bool = True,
	judges=None,
	extend_system: str | None = None,
	first: int = 0,
) -> Path:
	"""Reps [first, reps) of every task; done (task, rep) pairs are skipped, so a rerun resumes."""

	async def one(t, rep, vdir):
		if t.get('context'):
			return await run_case_task(t, rep, model, judges, extend_system)
		return await run_case(t, rep, model, vdir, max_steps, headless, judges, extend_system)

	return await run_eval(tasks, one, flow, variant, reps, timeout_s, concurrency, first)
