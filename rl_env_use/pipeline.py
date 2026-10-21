"""The pipeline: browser agents mine the sources where the topic's data lives, you look at the data, then grounded
tasks are written and verified.
build = scout (cases / quote-verified facts) -> look (what plain HTTP would have seen) -> [you review the data] -> synth -> verify [-> panel].
create = the same stages with no human in the loop: a planner writes the spec from one line, then mining rounds
continue until the env holds enough verified tasks with headroom, or the budget is spent.
With a `skill` in the spec the policy reads each case's materials and answers with no browser (the env tests the domain);
without one it is a browser agent answering on the live web (the env tests browsing)."""

import asyncio
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import ui
from .browse import host, http_capture, quote_in
from .env import Env
from .graders import grade_answer
from .llm import LEDGER, call
from .rollout import rollout, trace_grade
from .scout import scout
from .store import load_jsonl, save_jsonl, write_atomic
from .tasks import balanced, calibration, cases, keep, rejected_facts, synthesize, synthesize_cases
from .verify import ADJUDICATORS, PANEL, cap_all_fail, check_live, closed_book, critique, rescore_panel
from .verify import panel as run_panel

STAGES = ('scout', 'look', 'synth', 'verify', 'panel', 'harden')


async def look(env: Path) -> str:
	"""Data first: what was mined, from where, and how much of it a fetch tool would have seen (the measured edge)."""
	kb = json.loads((env / 'kb.json').read_text())
	items = kb['facts'] + [m | {'quote': m['text'][:200]} for m in kb.get('materials', [])]  # a material counts as seen if its opening is
	todo = list(dict.fromkeys(f['url'] for f in items if 'http' not in f))
	sem = asyncio.Semaphore(8)

	async def fetch(u):
		async with sem:
			return u, await http_capture(u)

	pages = dict(await asyncio.gather(*(fetch(u) for u in todo)))
	for f, orig in zip(items, kb['facts'] + kb.get('materials', [])):
		if f['url'] in pages:
			p = pages[f['url']]
			orig['http'] = 'visible' if quote_in(f['quote'], p['text']) else 'in_raw_html' if quote_in(f['quote'], p['raw']) else 'blocked' if p['blocked'] else 'missing'
	(env / 'kb.json').write_text(json.dumps(kb, indent=1))
	if kb.get('materials'):
		cs = cases(kb)
		mats = kb['materials']
		hidden = sum(m.get('http') not in ('visible', None) for m in mats)
		head = f'look: {len(cs)} complete cases (materials + outcome) from {len(mats)} materials on {len({host(m["url"]) for m in mats})} sites; {len({m["case"] for m in mats}) - len(cs)} cases still lack an outcome'
		head += f'\n  plain HTTP fetch of the material pages: {hidden}/{len(mats)} materials not visible without a browser'
	else:
		head = None
	n, http = len(kb['facts']), Counter(f.get('http') for f in kb['facts'])
	sites = Counter(host(f['url']) for f in kb['facts'])
	errs = [m for m in kb['missions'] if m.get('error')]
	lines = ([head] if head else []) + [
		f'{"  " if head else "look: "}{n} quote-verified facts from {len({f["url"] for f in kb["facts"]})} pages on {len(sites)} sites: '
		+ ', '.join(f'{h} {c}' for h, c in sites.most_common(8))
	]
	if n:
		lines.append(
			f'  plain HTTP fetch of the same pages: {http["visible"] / n:.0%} visible, {http["in_raw_html"] / n:.0%} only in raw HTML, {(http["blocked"] + http["missing"]) / n:.0%} unreachable without a browser'
		)
	lines += [f'  mission failed: {m["start_url"]} {m["error"][:120]}' for m in errs]
	lines += [f'  scout note: {o[:200]}' for m in kb['missions'] for o in m.get('observations', [])[:1]][:6]
	return '\n'.join(lines)


def kb_for_synth(env: Path) -> dict:
	"""The mined data minus the facts you rejected in `review` (Data tab)."""
	kb = json.loads((env / 'kb.json').read_text())
	bad = rejected_facts(env)
	return kb | {'facts': [f for f in kb['facts'] if f['id'] not in bad], 'materials': [m for m in kb.get('materials', []) if m['id'] not in bad]}


def use_browser_spec(spec: dict) -> None:
	"""The env's proxy (Chromium --proxy-server) for every browser it opens: miners, checks and policies alike."""
	if spec.get('proxy'):
		os.environ['RLENV_PROXY'] = spec['proxy']


def unverified(t: dict) -> bool:
	"""Checks missing or errored (API/browser hiccup) - re-run, never mistaken for a real failure."""
	c = t['checks']
	ok = lambda v: isinstance(v, bool) or v == 'skipped'  # noqa: E731
	cb = c.get('closed_book', {None: None})
	return c.get('critic') is None or 'critic error' in str(c['critic']) or not ok(c.get('live')) or not (cb == 'skipped' or all(map(ok, cb.values())))


async def build(env: Path, spec: dict, only: str | None = None, model: str = 'sonnet46', **_) -> str:
	use_browser_spec(spec)
	out, scouted = [], False
	if only in (None, 'scout') and (only or not (env / 'kb.json').exists()):
		kb = await scout(
			spec['topic'],
			env,
			spec.get('missions', 4),
			spec.get('agent_model', 'sonnet46'),
			spec.get('max_steps', 40),
			True,
			spec.get('concurrency', 4),
			model,
			spec.get('angles'),
			spec.get('sources'),
			spec.get('skill'),
			spec.get('enrich', False),
		)
		out.append(f'scout: {len(cases(kb))} complete cases, {len(kb.get("materials", []))} materials, {len(kb["facts"])} quote-verified facts from {len(kb["missions"])} missions')
		scouted = True
	if only in (None, 'look') and (only or scouted or any('http' not in f for f in json.loads((env / 'kb.json').read_text())['facts'])):
		out.append(await look(env))
	if scouted and not only:
		return '\n'.join(out + [f'Look at the mined data before any task is written: rl-env-use review {env} (Data tab, reject bad facts). Then run build again.'])
	if only in (None, 'synth') and (only or not (env / 'tasks.jsonl').exists()):
		out.append(await synth_stage(env, spec, model))
	if only in (None, 'verify'):
		out.append(await verify_stage(env, spec, model))
	if only == 'panel' or (only in (None, 'verify') and spec.get('panel')):
		out.append(await panel_stage(env, spec))
	if only == 'harden':  # too-easy tasks -> harder ones on the same cases, checked and measured like any draft
		out += [await harden_stage(env, spec, model), await verify_stage(env, spec, model), await panel_stage(env, spec)]
	return '\n'.join(out)


async def synth_stage(env: Path, spec: dict, model: str) -> str:
	kb = kb_for_synth(env) | {'skill': spec.get('skill', '')}
	old = {t['id']: t for t in load_jsonl(env / 'tasks.jsonl')}
	if spec.get('skill'):  # another synth round tops every case up to tasks_per_case
		new = await synthesize_cases(
			kb,
			spec.get('tasks_per_case', 2),
			model,
			[t['prompt'] for t in old.values()],
			spec.get('concurrency', 4),
			Counter(t.get('case') for t in old.values()),
			spec.get('hard', False),
			spec.get('mix', 'mixed'),
			spec.get('conventions', ''),
			calibration(list(old.values())),
		)
	else:
		new = await synthesize(kb, spec.get('n_tasks', 30), model, [t['prompt'] for t in old.values()], spec.get('hard', False))
	old.update({t['id']: t for t in new if t['id'] not in old})
	save_jsonl(env / 'tasks.jsonl', list(old.values()))
	return f'synth: {len(new)} tasks drafted'


async def verify_stage(env: Path, spec: dict, model: str) -> str:
	tasks = load_jsonl(env / 'tasks.jsonl')
	kb = json.loads((env / 'kb.json').read_text())
	facts = {f['id']: f for f in kb['facts']}
	todo = [t for t in tasks if not t['checks']['structural'] and not t['checks']['leakage'] and unverified(t)]
	await closed_book(todo, spec.get('closed_book', ['haiku', 'sonnet46']))
	if spec.get('live_check', True):
		await check_live(todo, facts)
	else:  # miners verified every quote on the live page when recording it; the re-check (a Modal browser) is for staleness
		for t in todo:
			t['checks']['live'] = 'skipped'
	await critique(todo, facts, model)
	mats = {m['id']: m for m in kb.get('materials', [])}
	for t in tasks:  # the edge per task: needs_browser = some evidence or material is invisible to a plain fetch
		t['needs_browser'] = any(facts.get(f, {}).get('http') not in ('visible', None) for f in t['fact_ids']) or any(
			mats.get(m, {}).get('http') not in ('visible', None) for m in t.get('material_ids', [])
		)
	save_jsonl(env / 'tasks.jsonl', tasks)
	kept = [t for t in tasks if keep(t)]
	return f'verify: {len(kept)}/{len(tasks)} tasks pass all checks; {sum(t["needs_browser"] for t in kept)} of them rest on data a plain fetch does not see'


async def harden_stage(env: Path, spec: dict, model: str) -> str:
	"""Every task the whole panel solved (or that asked for several values) goes back to the drafter with the env's
	calibration: one harder task on the same case per such task, up to `harden_levels` deep. Difficulty is a design goal,
	not a filter's leftovers."""
	tasks = load_jsonl(env / 'tasks.jsonl')
	easy = {}
	for t in tasks:
		p = t['checks'].get('panel', {})
		if p.get('verdict') in ('saturated', 'multi_value') and not p.get('hardened') and t.get('level', 0) < spec.get('harden_levels', 2):
			easy.setdefault(t['case'], []).append(t)
			p['hardened'] = True
	if not easy:
		return 'harden: no too-easy tasks left to harden'
	kb = kb_for_synth(env) | {'skill': spec.get('skill', '')}
	new = await synthesize_cases(
		kb,
		model=model,
		existing=[t['prompt'] for t in tasks],
		concurrency=spec.get('concurrency', 4),
		mix=spec.get('mix', 'mixed'),
		conventions=spec.get('conventions', ''),
		calibration=calibration(tasks),
		easy=easy,
	)
	ids = {t['id'] for t in tasks}
	save_jsonl(env / 'tasks.jsonl', tasks + [t for t in new if t['id'] not in ids])
	return f'harden: {sum(map(len, easy.values()))} too-easy tasks on {len(easy)} cases -> {len(new)} harder drafts'


def panel_todo(t: dict, panel: list[str] | None = None) -> bool:
	"""Passes every correctness gate and has no difficulty measurement yet (or an errored one)."""
	p = t['checks'].get('panel', {})
	stale = panel is not None and 'verdict' in p and p.get('models') != list(panel)  # measured by another panel
	return keep(t | {'checks': t['checks'] | {'panel': {'verdict': 'keep'}}}) and (p.get('verdict', 'unmeasured') == 'unmeasured' or stale)


async def panel_stage(env: Path, spec: dict) -> str:
	tasks = load_jsonl(env / 'tasks.jsonl')
	facts = {f['id']: f for f in json.loads((env / 'kb.json').read_text())['facts']}
	for t in tasks:
		rescore_panel(t)
	todo = [t for t in tasks if t.get('context') and panel_todo(t, spec.get('panel') or PANEL)]  # a new panel re-measures
	await run_panel(todo, facts, spec.get('panel') or PANEL, spec.get('panel_reps', 2), spec.get('adjudicators') or ADJUDICATORS)
	capped = cap_all_fail(tasks, spec.get('all_fail_cap', 0.2))
	save_jsonl(env / 'tasks.jsonl', tasks)
	v = Counter(t['checks']['panel']['verdict'] for t in todo)
	return f'panel: {len(todo)} tasks measured on {", ".join(spec.get("panel") or PANEL)}: {dict(v)}' + (f'; {capped} all-fail tasks over the cap' if capped else '')


def measured(t: dict) -> bool:
	return t['checks'].get('panel', {}).get('verdict', 'unmeasured') != 'unmeasured'


def panel_means(tasks: list[dict], kind: str | None = None, kept: bool = True) -> dict[str, float]:
	"""Mean score per panel model over kept tasks (of one kind): the env's difficulty, weak to strong. kept=False: over
	every measured task, saturated ones included - the headroom of what the drafter writes."""
	acc = {}
	for t in tasks:
		if (keep(t) if kept else measured(t)) and (kind is None or t.get('kind') == kind):
			for m, x in t['checks'].get('panel', {}).get('scores', {}).items():
				acc.setdefault(m, []).append(x)
	return {m: round(sum(v) / len(v), 3) for m, v in acc.items()}


def panel_lift(tasks: list[dict]) -> float | None:
	"""Kept forecasts: the panel's mean score with the materials minus without - do the materials help on average?"""
	d = [
		(sum(p['scores'].values()) / len(p['scores'])) - p['blind_score']
		for t in tasks
		if keep(t) and t['answer_type'] == 'prob'
		for p in [t['checks']['panel']]
		if p.get('blind_score') is not None
	]
	return round(sum(d) / len(d), 4) if d else None


class SpecPlan(BaseModel):
	name: str = Field(description='Short kebab-case env name, e.g. sec-credit or trial-forecast.')
	topic: str = Field(
		description='The domain and which cases to collect (time window, kind of entity), in 1-3 sentences. Prefer lesser-known, recent cases: famous ones are memorized.'
	)
	skill: str = Field(
		description='What the model must do, input -> outcome: what it is given (the materials) and what it must produce. The outcome must be published later than or apart from the input.'
	)
	sources: list[str] = Field(description='2-6 real https URLs where the inputs and outcomes live: official registries, databases, filings search pages. Never a search engine.')
	mix: Literal['derive', 'predict', 'mixed'] = Field(
		description='derive: compute or diagnose from the materials; predict: forecast a later outcome the materials bear on; mixed: both.'
	)
	conventions: str = Field(
		description='The canonical definitions an expert in this domain uses for the quantities the tasks will ask about (one line each), so every task can state its definition.'
	)
	sampling: str = Field(
		description='How cases are chosen WITHOUT looking at their outcome (e.g. all filings of a sector in a window), so the env is not biased toward cases with notable outcomes.'
	)


PLAN_SYS = """You design an RL environment that trains or measures a domain skill from real cases mined from the web by browser agents.
A case = materials the model is given + the true outcome, published later or elsewhere. The model answers offline from the materials.
From the user's one-line request, write the spec. Good skills pair an input document with an outcome that is published later or elsewhere and is
precise (a number, a category, a yes/no event with a threshold). Avoid outcomes driven by events the materials cannot foresee. If the user names
sources, use them; otherwise pick the official primary sources where both inputs and outcomes are public without login."""


async def plan_spec(line: str, model: str = 'sonnet46') -> dict:
	p, _, _ = await call(model, PLAN_SYS, f'REQUEST: {line}', SpecPlan)
	return p.model_dump() | {'sources': [u for u in p.sources if re.fullmatch(r'https?://[^\s{}<>]+', u)]}  # a template like /study/{id} is no start page


def show(tasks: list[dict], target: int) -> None:
	"""The board's view of the env: kept tasks against the target, the forecast outcome balance, the panel's scores."""
	kept = balanced([t for t in tasks if keep(t)])
	fc = Counter(t['answer'] for t in kept if t.get('answer_type') == 'prob')
	kinds = Counter(t.get('kind') for t in kept)
	ui.line('tasks', f'{len(kept)}/{target} kept ({kinds["derive"]} derive, {kinds["predict"]} forecast: {fc["yes"]} yes / {fc["no"]} no) of {len(tasks)} drafted')
	if m := panel_means(tasks):
		ui.line('panel', '  '.join(f'{k} {v:.2f}' for k, v in m.items()) + '  (mean score on kept tasks)')
	ui.progress(len(kept) / max(target, 1))


async def create(env: Path, spec: dict, model: str = 'sonnet46', log=print) -> dict:
	"""Mining rounds until `target_tasks` kept tasks, the budget, or two rounds that add (almost) nothing. Each round: scout
	(cases first finish their outcomes) -> synth top-up -> verify -> panel. Headroom steers the next round: when the
	strongest panel model saturates the derive tasks, synth moves to forecasts."""
	use_browser_spec(spec)
	target, dry, prior = spec.get('target_tasks', 150), 0, spec.get('spent_usd', 0.0)  # a resumed env keeps its earlier spend
	budget = spec.get('budget_usd', 40.0) - prior
	per_mission = 0.35  # $ per mining mission (agent tokens + its browser), measured on earlier envs
	missions = spec.get('missions', 12)

	def save_spend():  # after every stage: a crash or kill must not forget what was spent
		spec['spent_usd'] = round(prior + LEDGER.usd, 4)
		write_atomic(env / 'spec.json', json.dumps(spec, indent=1))

	ui.money(lambda: prior + LEDGER.usd, spec.get('budget_usd', 40.0))
	show(load_jsonl(env / 'tasks.jsonl'), target)
	for rnd in range(1, spec.get('max_rounds', 6) + 1):
		ui.line('round', f'{rnd} of at most {spec.get("max_rounds", 6)}')
		left = budget - LEDGER.usd
		missions = min(missions, int(left * 0.5 / per_mission))  # mining gets at most half of what is left
		if missions < 1:
			log(f'round {rnd}: budget spent (${LEDGER.usd:.2f} of ${budget:.0f})')
			break
		before = len(balanced([t for t in load_jsonl(env / 'tasks.jsonl') if keep(t)]))
		topic = spec['topic'] + (f'\nCASE SELECTION (never by outcome): {spec["sampling"]}' if spec.get('sampling') else '')
		kb = await scout(
			topic,
			env,
			missions,
			spec.get('agent_model', 'sonnet46'),
			spec.get('max_steps', 60),
			True,
			spec.get('concurrency', 6),
			model,
			None,
			spec.get('sources'),
			spec.get('skill'),
		)
		log(f'round {rnd}: scout {missions} missions -> {len(cases(kb))} complete cases (${prior + LEDGER.usd:.2f} spent in total)')
		ui.stage('look', 1)
		log(f'round {rnd}: ' + (await look(env)).replace('\n', f'\nround {rnd}: '))  # free: what a plain HTTP fetch of the same pages shows
		ui.tick()
		save_spend()
		for stage in (synth_stage, verify_stage, panel_stage, harden_stage, verify_stage, panel_stage):  # every round also hardens what the panel found too easy
			if LEDGER.usd >= budget:  # unmeasured drafts wait for a resumed run with more budget
				log(f'round {rnd}: budget spent (${LEDGER.usd:.2f} of ${budget:.0f}) - skipping {stage.__name__} and the rest of the round')
				break
			log(f'round {rnd}: ' + await (stage(env, spec) if stage is panel_stage else stage(env, spec, model)))
			save_spend()
		tasks = load_jsonl(env / 'tasks.jsonl')
		kept = balanced([t for t in tasks if keep(t)])
		gained, yield_ = len(kept) - before, len(kept) / max(1, len(cases(kb)))
		means = panel_means(tasks, 'derive', kept=False)
		log(
			f'round {rnd}: {len(kept)}/{target} tasks kept (+{gained}), {yield_:.2f} per case; panel means on all drafted derive tasks {means}; ${prior + LEDGER.usd:.2f} spent in total'
		)
		show(tasks, target)
		if len(kept) >= target:
			break
		dry = dry + 1 if gained < 0.1 * missions else 0
		if dry >= 2:
			log(f'round {rnd}: two rounds without new tasks - stopping')
			break
		if spec.get('mix') != 'predict' and sum(t.get('kind') == 'derive' and measured(t) for t in tasks) >= 20 and means and max(means.values()) > 0.85:
			spec['mix'] = 'predict'
			log(f'round {rnd}: the strongest model scores {max(means.values()):.0%} on derive tasks - next rounds write forecasts')
			save_spend()
		missions = math.ceil((target - len(kept)) / max(yield_, 0.5) / 3 * 1.3)
	tasks = load_jsonl(env / 'tasks.jsonl')
	kept = balanced([t for t in tasks if keep(t)])
	spec['result'] = {
		'kept': len(kept),
		'cases': len({t.get('case') for t in kept}),
		'kinds': dict(Counter(t.get('kind') for t in kept)),
		'panel_means': panel_means(tasks),
		'verdicts': dict(Counter(t['checks'].get('panel', {}).get('verdict', '-') for t in tasks)),
		'forecast_lift': panel_lift(tasks),
		'usd': round(prior + LEDGER.usd, 2),
		'status': 'ready' if len(kept) >= min(target, 30) else 'incomplete',
	}
	save_spend()
	return spec['result']


async def run(env: Path, spec: dict, a) -> Path:
	use_browser_spec(spec)
	tasks = Env(env, verified=getattr(a, 'verified', False)).tasks()
	tasks = tasks[: a.limit] if a.limit else tasks
	ext = Path(a.system).read_text() if a.system else None
	if getattr(a, 'blind', False):  # ablation: same tasks, materials withheld - kept out of the hillclimb flow
		tasks = [t | {'context': '(materials withheld)'} for t in tasks if t.get('context')]
	flow = getattr(a, 'flow', None) or ('ablation' if getattr(a, 'blind', False) else 'hillclimb')
	return await rollout(tasks, env / flow, a.variant, a.model, a.reps, spec.get('max_steps_policy', 25), a.timeout, a.concurrency, True, None, ext)


def load_kept(env: Path, verified: bool = False) -> list[dict]:
	return [t | {'category': t.get('category', '')} for t in Env(env, verified=verified).tasks()]


def regrade_row(vdir: Path, task: dict, row: dict) -> dict:
	tp = vdir / 'traces' / f'{row["prompt_id"]}_rep{row["rep"]}.json'
	correct = float(grade_answer(task, row['meta']['answer']))
	row['meta']['gold'] = task['answer']
	return trace_grade(task, correct, json.loads(tp.read_text())) if tp.exists() and not task.get('context') else {'correct': correct}


def export(env: Path, out: Path) -> int:
	return Env(env).export_jsonl(out)
