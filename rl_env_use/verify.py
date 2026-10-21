"""Model and browser checks on drafted tasks, recorded on each task under `checks` (structural and leakage: checks.py).

closed_book  can a model answer from memory alone? (memorized outcomes measure recall, not skill)
live         every cited quote is still present on its source page (deterministic, re-runnable for staleness)
critic       independent LLM check: one unambiguous answer, gold follows from evidence (+ case tasks: the context
             does not state the answer, and the task is well posed)
panel        difficulty, measured apart from correctness: models from several labs solve it with the materials.
             All-pass tasks are dropped, all-fail or converging-wrong ones need an adjudicator to confirm the gold,
             forecasts must move with the materials and not be answerable without them (judged blind to the outcome)
"""

import asyncio
import math
import re

from pydantic import BaseModel, Field

from . import ui
from .browse import browser, capture, quote_in
from .checks import grade_answer
from .graders import norm, numbers, prob_of
from .llm import call, now
from .tasks import keep


class Recompute(BaseModel):
	calculation: str = Field(description='The values you read from the evidence and the arithmetic you applied, if any (day differences: exclusive, end minus start).')
	recomputed_answer: str = Field(description='ONLY the bare final answer (a single number, date, name or comma-separated list) - no units, words or arithmetic.')


class Critique(BaseModel):
	unambiguous: bool = Field(description='The prompt asks for exactly one value, and exactly one correct answer exists for it.')
	gold_supported: bool = Field(description='The gold answer follows from the evidence quotes.')
	realistic: bool = Field(description='A real person might plausibly need this.')
	issues: str


class CaseCritique(Critique):
	answer_in_context: bool = Field(
		description='The final answer itself is written in the context (or a trivial rewording of it), so no expertise is needed. Inputs from which it must be computed or inferred do not count.'
	)
	well_posed: bool = Field(
		description='derive: the context holds everything an expert needs to reach exactly the gold. predict: the gold is a real outcome the context bears on but neither the context nor the prompt reveals.'
	)


class ClosedBook(BaseModel):
	answer: str


def guessable(t: dict) -> bool:
	"""A case task with a handful of candidates: a closed-book hit is chance, not memory."""
	return bool(t.get('context')) and (t['answer_type'] == 'prob' or bool(t.get('choices')) or norm(t['answer']) in ('yes', 'no', 'true', 'false'))


async def closed_book(tasks: list[dict], models: list[str]) -> None:
	"""The prompt alone, no materials and no web: a right answer means the outcome is memorized."""
	sys = 'Answer from memory only. Give your single best final answer, no hedging.'
	for t in tasks:
		if guessable(t):
			t['checks']['closed_book'] = 'skipped'
	tasks = [t for t in tasks if not guessable(t)]

	async def one(t, m):
		try:
			r, _, _ = await call(m, sys, t['prompt'], ClosedBook)
			t['checks'].setdefault('closed_book', {})[m] = grade_answer(t, r.answer)
			t['checks'].setdefault('closed_book_answers', {})[m] = r.answer
		except Exception as e:
			t['checks'].setdefault('closed_book', {})[m] = f'error: {e}'

	await ui.gather((one(t, m) for t in tasks for m in models), 'closed-book')


async def check_live(tasks: list[dict], facts: dict, headless: bool = True) -> None:
	"""Re-open every cited source page in a real browser and confirm each quote is still there."""
	session = await browser().start()
	cache = {}
	try:
		for t in tasks:
			ok = True
			for fid in t['fact_ids']:
				f = facts.get(fid)
				if not f:
					ok = False
					continue
				for _attempt in range(2) if f['url'] not in cache else ():
					cache[f['url']] = None
					try:
						await session.goto(f['url'])
						for _ in range(5):  # dynamic pages (GitHub, HF, arXiv HTML) render late: poll before calling it stale
							await asyncio.sleep(2)
							cache[f['url']] = await session.text()
							if quote_in(f['quote'], cache[f['url']]):
								break
						else:  # lazy sections (long database records): expand and scroll once more before calling it stale
							cache[f['url']] = (await capture(session, f['url']))['text']
						break
					except Exception:  # a dead browser would fail every later page: start a fresh one and retry
						await session.kill()
						session = await browser().start()
				text = cache[f['url']]
				ok = None if text is None else ok and quote_in(f['quote'], text)  # None = page never loaded: retried, not stale
				if ok is None:
					break
			t['checks']['live'] = ok
			t['checks']['live_at'] = now()
	finally:
		await session.kill()


CASE_CRITIC = (
	' The model under test reads the CONTEXT and answers the PROMPT; the EVIDENCE is the true outcome it never sees. Flag tasks the context answers outright, and tasks that are not well posed.'
	' A predict task is uncertain by design, so judge it as a forecasting question: unambiguous = exactly one outcome resolves it (not: one forecast is forced);'
	' well posed = the predicted quantity is precisely defined, the evidence resolves it, and the context gives an expert a real basis to forecast it (design, size, arms, trends) -'
	' not well posed if it hinges on events the context cannot foresee (a financing, an acquisition, an outside halt),'
	' or if the PROMPT states, paraphrases or hints at the outcome ("the posted results show...") or points to results data the context does not hold.'
)


async def critique(tasks: list[dict], facts: dict, model: str = 'sonnet46') -> None:
	sys = 'You audit benchmark tasks. Be strict: flag any prompt that admits more than one reasonable answer, or any gold not supported by the evidence.'

	async def one(t):
		ev = '\n'.join(f'"{facts[f]["quote"]}" ({facts[f]["url"]})' for f in t['fact_ids'] if f in facts)
		try:  # blind recomputation first: a critic that sees the gold mostly copies it
			if t.get('kind') == 'derive':  # well posed only if the gold follows from what the model is given: the materials alone
				r, _, _ = await call(
					model,
					'Answer the question from the materials only. If they lack what the answer needs, say exactly what is missing instead of guessing.',
					f'QUESTION: {t["prompt"]}\nMATERIALS:\n{t["context"]}',
					Recompute,
				)
			else:
				how = (
					('The EVIDENCE states the actual outcome: report it, do not forecast.' + (' Answer yes or no: did it happen?' if t['answer_type'] == 'prob' else ''))
					if t.get('kind') == 'predict'
					else ''
				)
				r, _, _ = await call(model, f'Answer the question from the evidence only. {how}'.strip(), f'QUESTION: {t["prompt"]}\nEVIDENCE:\n{ev}', Recompute)
			calc = f'; expression evaluated by code: {t["expression"]}' if t.get('expression') else ''
			user = f'PROMPT: {t["prompt"]}\nGOLD: {t["answer"]} (type {t["answer_type"]}; derivation: {t.get("derivation") or "literal"}{calc})\nEVIDENCE:\n{ev}'
			if t.get('context'):  # case task: the model under test sees CONTEXT + PROMPT; the evidence is the outcome it never sees
				user = f'CONTEXT GIVEN TO THE MODEL ({t["kind"]} task):\n{t["context"]}\n\n{user}'
			c, _, _ = await call(model, sys + CASE_CRITIC if t.get('context') else sys, user, CaseCritique if t.get('context') else Critique)
			t['checks']['critic'] = c.model_dump() | r.model_dump() | {'agrees': grade_answer(t, r.recomputed_answer) >= 0.5}
		except Exception as e:  # unverified = rejected
			t['checks']['critic'] = {'unambiguous': False, 'gold_supported': False, 'agrees': False, 'issues': f'critic error: {e}'[:300]}

	await ui.gather((one(t) for t in tasks), 'critic')


class Adjudication(BaseModel):
	gold_correct: bool = Field(description='The gold follows from the materials (derive) or from the outcome evidence (predict) under the definitions the prompt states.')
	ambiguous: bool = Field(description='A reasonable expert reading of the prompt leads to a different answer: a basis, period or definition the prompt does not rule out.')
	panel_error: str = Field(description="The concrete mistake behind the models' answer - or why they are right and the gold is wrong.")


PANEL = ['opus55', 'gpt-astra']  # frontier models of two labs (your Claude and ChatGPT accounts): a task is dropped only when both solve it


ADJUDICATORS = ['gpt-astra', 'opus55']  # the first one that did not give the disputed answer


SHIFT = 0.1  # forecasts: the materials must move the panel's probability by this much (outcome-blind: surprises stay in)


MEMORIZED = 0.96  # forecasts: a panel this right without the materials knows the outcome (1-Brier: p >= 0.8 on the true side; a 0.7 lean scores 0.91)


def format_misses(t: dict, answers: list) -> list[str]:
	"""Answers graded wrong only because they state several values, one of them the gold ('48,520 thousand, 201.3%'): the
	prompt invited more than one answer. A task design defect - never difficulty, though it fails strong models."""
	if t['answer_type'] != 'number':
		return []
	parts = lambda a: [x for x in re.split(r'[;,]\s+|\s+and\s+|\n', a) if x.strip()]  # noqa: E731
	return [a for a in answers if a and not grade_answer(t, a) and len(parts(a)) > 1 and any(grade_answer(t, x) for x in parts(a))]


def answer_key(t: dict, a: str) -> str:
	n = numbers(a)
	return f'{n[0]:.4g}' if t['answer_type'] == 'number' and len(n) == 1 else norm(a)


async def panel(tasks: list[dict], facts: dict, models: list[str] = PANEL, k: int = 2, adjudicators: list[str] = ADJUDICATORS) -> None:
	"""Difficulty, measured apart from correctness. Each model answers k times with the materials (forecasts: once more
	without them). All-pass tasks teach nothing; all-fail tasks and wrong answers two models share are where wrong golds
	hide (HLE-style filtering), so an adjudicator that did not give that answer must confirm the gold and name the error.
	Forecasts are judged without looking at the outcome: kept if the materials move the panel's probability and the
	panel cannot answer blind; whether the materials help on average is an env-level number (panel_lift)."""
	from .rollout import public, solve_case

	async def solve(t, m, blind=False):
		pub = public(t) | ({'context': '(materials withheld)'} if blind else {})
		try:
			return (await solve_case(pub, m))[2]
		except Exception:  # a refusal or outage is no answer, not a wrong one
			return None

	async def one(t):
		jobs = [(m, False) for m in models for _ in range(k)] + ([(m, True) for m in models] if t['answer_type'] == 'prob' else [])
		got = await asyncio.gather(*(solve(t, m, b) for m, b in jobs))
		p = {'answers': {}, 'blind': {}, 'models': list(models)}
		for (m, b), a in zip(jobs, got):
			p['blind' if b else 'answers'].setdefault(m, []).append(a)
		sc = {m: [float(grade_answer(t, a)) for a in v if a is not None] for m, v in p['answers'].items()}
		sc = {m: v for m, v in sc.items() if v}
		p['scores'] = {m: sum(v) / len(v) for m, v in sc.items()}
		flat = [x for v in sc.values() for x in v]
		if len(sc) < 2:
			p['verdict'] = 'unmeasured'
		elif t['answer_type'] == 'prob':
			forecast_verdict(t, p, flat)
		elif fm := format_misses(t, [a for v in p['answers'].values() for a in v]):
			p['verdict'], p['format_misses'] = 'multi_value', fm[:3]
		elif min(flat) >= 1:
			p['verdict'] = 'saturated'
		else:
			wrong = {}
			for m, v in p['answers'].items():
				for a in v:
					if a is not None and not grade_answer(t, a):
						wrong.setdefault(answer_key(t, a), set()).add(m)
			shared = [w for w, ms in wrong.items() if len(ms) >= 2]
			p['all_fail'] = max(flat) == 0
			p['verdict'] = 'keep'
			if p['all_fail'] or shared:
				modal = max(wrong, key=lambda w: len(wrong[w]))
				judge = next((m for m in adjudicators if m not in wrong[modal]), adjudicators[0])
				ev = '\n'.join(f'"{facts[f]["quote"]}"' for f in t['fact_ids'] if f in facts)
				calc = (
					f"\nTHE DRAFTER'S COMPUTATION (arithmetic checked by code; its inputs and formula are NOT verified): {t['expression']}"
					if t.get('expression')
					else f'\nDERIVATION: {t.get("derivation") or "literal"}'
				)
				try:
					j, _, _ = await call(
						judge,
						'You adjudicate a benchmark task that strong models answer differently from its gold. Work it out yourself from the materials and evidence first, then decide. Be strict about ambiguity.',
						f"MATERIALS GIVEN TO THE MODELS:\n{t['context']}\n\nQUESTION: {t['prompt']}\nGOLD: {t['answer']}{calc}\nOUTCOME EVIDENCE:\n{ev or '-'}\nMODELS' ANSWER: {modal} ({len(wrong[modal])} models)",
						Adjudication,
					)
					p['adjudication'] = j.model_dump() | {'model': judge}
					p['verdict'] = 'keep' if j.gold_correct and not j.ambiguous else 'gold_disputed'
				except Exception as e:
					p['verdict'], p['error'] = 'unmeasured', str(e)[:200]
		t['checks']['panel'] = p

	await ui.gather((one(t) for t in tasks), 'panel')


def forecast_verdict(t: dict, p: dict, flat: list[float]) -> None:
	"""Outcome-blind: do the materials move the panel's probability, and can the panel answer without them?"""
	mean_p = lambda v: (lambda ps: sum(ps) / len(ps) if ps else None)([x for x in map(prob_of, v) if x is not None])  # noqa: E731
	shifts = [abs(w - b) for m in p['answers'] if (w := mean_p(p['answers'].get(m, []))) is not None and (b := mean_p(p.get('blind', {}).get(m, []))) is not None]
	blind = [float(grade_answer(t, a)) for v in p.get('blind', {}).values() for a in v if a is not None]
	p['shift'] = round(sum(shifts) / len(shifts), 4) if shifts else None
	p['blind_score'] = round(sum(blind) / len(blind), 4) if blind else None
	p['verdict'] = (
		'unmeasured'
		if p['shift'] is None
		else 'memorized'
		if (p['blind_score'] or 0) >= MEMORIZED
		else 'no_shift'
		if p['shift'] < SHIFT
		else 'saturated'
		if min(flat) >= 0.95
		else 'keep'
	)


def rescore_panel(t: dict) -> None:
	"""Re-grade a task's stored panel answers with today's grader and thresholds: a grader fix must reach the difficulty
	verdicts too, both ways (a right answer with its working in parentheses once read as a miss and kept a task as 'hard';
	a 0.7 lean without the materials once read as memory). A task no longer saturated or multi-valued whose disagreement
	now needs an adjudicator is measured again."""
	p = t['checks'].get('panel')
	if not p or not p.get('answers') or p.get('verdict') not in ('keep', 'capped', 'saturated', 'multi_value', 'memorized', 'no_shift'):
		return
	sc = {m: [float(grade_answer(t, a)) for a in v if a is not None] for m, v in p['answers'].items()}
	sc = {m: v for m, v in sc.items() if v}
	if len(sc) < 2:
		return
	p['scores'] = {m: sum(v) / len(v) for m, v in sc.items()}
	flat = [x for v in sc.values() for x in v]
	if t['answer_type'] == 'prob':
		return forecast_verdict(t, p, flat)
	p['all_fail'] = max(flat) == 0
	if fm := format_misses(t, [a for v in p['answers'].values() for a in v]):
		p['verdict'], p['format_misses'] = 'multi_value', fm[:3]
	elif min(flat) >= 1:
		p['verdict'] = 'saturated'
	elif p['verdict'] in ('saturated', 'multi_value'):
		wrong = {}
		for m, v in p['answers'].items():
			for a in v:
				if a is not None and not grade_answer(t, a):
					wrong.setdefault(answer_key(t, a), set()).add(m)
		p.pop('format_misses', None)
		p['verdict'] = 'unmeasured' if p['all_fail'] or any(len(ms) >= 2 for ms in wrong.values()) else 'keep'


def cap_all_fail(tasks: list[dict], cap: float = 0.2) -> int:
	"""Even adjudicated, tasks no model solves concentrate wrong golds: at most `cap` of the kept set. Idempotent, and a
	capped task comes back when the env grows (the first ones by id are kept)."""
	verdict = lambda t: t['checks'].get('panel', {}).get('verdict')  # noqa: E731
	hard = sorted((t for t in tasks if t['checks'].get('panel', {}).get('all_fail') and verdict(t) in ('keep', 'capped')), key=lambda t: t['id'])
	for t in hard:
		t['checks']['panel']['verdict'] = 'keep'
	others = sum(keep(t) for t in tasks) - len([t for t in hard if keep(t)])
	allowed = math.floor(cap / (1 - cap) * others + 1e-9)
	extra = [t for t in hard if keep(t)][allowed:]
	for t in extra:
		t['checks']['panel']['verdict'] = 'capped'
	return len(extra)
