"""rl-env-use: an RL environment for a skill in any domain, built from real cases that browser agents mine where they live
(headless Chromium on Modal, driven through browser-harness-js; models through your Claude and ChatGPT CLIs or Modal GPUs).

autonomous:
  create  "<topic, skill, sources - one line>"   plan the spec, mine, write, verify and calibrate tasks until the env is ready
  climb   <env> --model M        hillclimb M's system prompt on the env (train/test split, paired CIs, keep only test gains)

step by step:
  new     <name> --topic T --skill S --sources URL..   the domain, what the model must do (input -> outcome), where the data lives
  build   <env>                 agents mine cases -> you look at the data -> build again: tasks are written and verified
  review  <env>                 local app: the mined data, the tasks, the policy traces
  run     <env> --model M       the model solves the tasks (reads each case, no browser); graded, audited, hillclimb/report.html
  report  <env>                 regrade stored runs (+ calibrated judges), audit, report.html
  compare <env>                 A/B against vanilla Claude Code browsing (WebSearch + WebFetch): mining and solving
  stop                          stop every Modal container of rl-env-use still running (each command stops its own)

advanced: codify (failure taxonomy from your notes), judges (narrow judges calibrated on your labels),
          split (train/test for /claude-api hillclimb), export (JSONL dataset)
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from . import ui
from .llm import LEDGER, load_dotenv
from .store import load_jsonl, save_jsonl

REPORT_BUILDER = Path(__file__).with_name('report') / 'build-report-lite.mjs'
METRICS = [('supported', 'Supported'), ('correct', 'Correct'), ('grounded', 'Grounded'), ('no_loop', 'No loop')]  # browse tasks
CASE_METRICS = [('correct', 'Correct')]


def run(coro):
	"""asyncio.run() without its shutdown phase (a CLI agent's pipes can outlive their task). Ctrl-C or a kill cancels the
	work and waits up to 45 s, so every agent's finally block stops its browser; then every Modal container still alive
	(model servers, anything missed) is stopped before the command returns."""
	from . import cloud

	loop = asyncio.new_event_loop()
	main = loop.create_task(coro)
	try:
		return loop.run_until_complete(main)
	except KeyboardInterrupt:
		main.cancel()
		loop.run_until_complete(asyncio.wait([main], timeout=45))
		raise
	finally:
		loop.run_until_complete(cloud.shutdown())


def spec_of(env: Path) -> dict:
	p = env / 'spec.json'
	if not p.exists():
		raise SystemExit(f'{p} not found - create the env with `new` first')
	return json.loads(p.read_text())


def flows(env: Path) -> list[Path]:
	"""Every run set under the env (hillclimb, models, ablation, ...): a directory holding <variant>/results.jsonl."""
	return sorted(d for d in env.iterdir() if d.is_dir() and any(d.glob('*/results.jsonl'))) if env.exists() else []


def write_state(env: Path, judges=(), flow: str = 'hillclimb') -> None:
	p = env / flow / '_state.json'
	s = json.loads(p.read_text()) if p.exists() else {}
	skill = (env / 'spec.json').exists() and json.loads((env / 'spec.json').read_text()).get('skill')
	metrics = [{'id': i, 'kind': 'binary', 'label': lab} for i, lab in (CASE_METRICS if skill else METRICS)]
	metrics += [{'id': j.id, 'kind': 'binary', 'label': j.id[:14]} for j in judges]
	ids = {m['id'] for m in metrics}
	s['metrics'] = metrics + [m for m in s.get('metrics', []) if m['id'] not in ids]  # headline = first binary metric
	s.setdefault('perf_fields', [{'id': 'cost_usd', 'label': 'Cost', 'unit': '$'}, {'id': 'latency_s', 'label': 'Latency', 'unit': 's'}, {'id': 'tool_calls', 'label': 'Actions'}])
	s.setdefault('goal', {'target': metrics[0]['id'], 'direction': 'higher', 'hold': ['cost_usd']})
	p.parent.mkdir(parents=True, exist_ok=True)
	p.write_text(json.dumps(s, indent=1))


def cmd_new(a):
	from .verify import PANEL

	env = Path('envs') / re.sub(r'[^a-z0-9]+', '-', a.name.lower()).strip('-')
	if (env / 'spec.json').exists():
		raise SystemExit(f'{env}/spec.json exists - edit it instead')
	env.mkdir(parents=True, exist_ok=True)
	spec = {'topic': a.topic, 'sources': a.sources, 'missions': max(4, 2 * len(a.sources)), 'agent_model': 'sonnet46', 'max_steps': 40, 'concurrency': 4}
	spec |= {'skill': a.skill, 'tasks_per_case': 2, 'panel': PANEL} if a.skill else {'n_tasks': 30, 'hard': False}
	spec |= {'proxy': a.proxy} if a.proxy else {}
	(env / 'spec.json').write_text(json.dumps(spec, indent=1))
	print(f'{env}/spec.json written. Next: rl-env-use build {env}')


def cmd_create(a):
	from . import pipeline
	from .verify import PANEL

	env = Path('envs') / re.sub(r'[^a-z0-9]+', '-', (a.name or '').lower()).strip('-') if a.name else None
	if env and (env / 'spec.json').exists():
		spec = spec_of(env)
		print(f'resuming {env} ({spec.get("result", {}).get("kept", 0)} tasks kept so far)')
	else:
		plan = run(pipeline.plan_spec(a.line, a.model))
		env = env or Path('envs') / re.sub(r'[^a-z0-9]+', '-', plan['name'].lower()).strip('-')
		if (env / 'spec.json').exists():
			raise SystemExit(f'{env} exists - pass --name {env.name} to resume it, or another --name')
		spec = {k: plan[k] for k in ('topic', 'skill', 'sources', 'mix', 'conventions', 'sampling')} | {
			'request': a.line,
			'auto': True,
			'missions': a.missions,
			'agent_model': 'sonnet46',
			'max_steps': 60,
			'concurrency': 6,
			'tasks_per_case': 4,
			'hard': True,
			'closed_book': ['haiku', 'sonnet46'],
			'panel': a.panel or PANEL,
			'panel_reps': 2,
		}
		print(json.dumps(spec, indent=1))
	spec |= {k: v for k, v in (('budget_usd', a.budget), ('target_tasks', a.tasks), ('mix', a.mix)) if v is not None}  # a resume keeps what it was given unless told otherwise
	spec.setdefault('budget_usd', 40.0)
	spec.setdefault('target_tasks', 150)
	if a.plan_only:
		return
	env.mkdir(parents=True, exist_ok=True)
	(env / 'spec.json').write_text(json.dumps(spec, indent=1))
	with ui.board(f'rl-env-use create · {env.name}'):
		res = run(pipeline.create(env, spec, a.model))
	print(json.dumps(res, indent=1))
	report(env, judges=False)
	print(f'next: rl-env-use climb {env} --model <the policy you will train>')


def cmd_climb(a):
	from .climb import climb
	from .pipeline import use_browser_spec

	env = Path(a.env)
	use_browser_spec(spec_of(env))
	from .judges import load_judges

	regrade(env, {j.id for j in load_judges(env / 'judges.json')})  # stored runs are scored with today's golds before any comparison; judge verdicts stay
	with ui.board(f'rl-env-use climb · {a.model} on {env.name}'):
		s = run(
			climb(
				env,
				a.model,
				a.rounds,
				a.k,
				a.reps,
				a.budget,
				a.proposer,
				Path(a.system).read_text() if a.system else '',
				a.test_frac,
				a.seed,
				a.confidence,
				a.mde,
				a.force,
				a.concurrency,
				a.verified,
			)
		)
	f = s.get('final', {})
	print(
		f'final: {f.get("decision")}'
		+ (
			f' - {f["variant"]}: test {f["baseline_test"]:.3f} -> {f["best_test"]:.3f}, paired {f["test"]["diff"]:+.3f} [{f["test"]["lo"]:+.3f}, {f["test"]["hi"]:+.3f}] on {f["reps"]} fresh reps after {f["test_looks"]} test looks'
			if 'test' in f
			else f' ({f.get("reason", "")})'
		)
	)
	report(env, judges=False)


def cmd_build(a):
	from . import pipeline

	if a.only and a.only not in pipeline.STAGES:
		raise SystemExit(f'--only must be one of {pipeline.STAGES}')
	print(run(pipeline.build(Path(a.env), spec_of(Path(a.env)) | ({'enrich': True} if a.enrich else {}), a.only, a.model)))


def cmd_review(a):
	from .annotate import serve

	srv = serve(Path(a.env), a.variant, a.port)
	print(f'review at http://127.0.0.1:{a.port}  (ctrl-c to stop)')
	try:
		srv.serve_forever()
	except KeyboardInterrupt:
		pass


def cmd_run(a):
	env = Path(a.env)
	a.model = a.model or 'sonnet46'
	if a.dry_run:  # the consent gate before a paid run: $ and minutes per episode measured on this env's earlier runs of the model
		from .audit import load_variants
		from .llm import MODELS, model_ids, resolve
		from .pipeline import load_kept

		n = len([t for t in load_kept(env, a.verified) if t.get('context') or not a.blind][: a.limit or None]) * a.reps  # the tasks the real run would take
		modal = any(r == 'modal' for r, _ in MODELS.get(a.model, [])) or a.model.startswith('modal:')
		# the same model measured on any route, earlier token-billed runs included - except for a Modal model, whose cost is GPU time
		mids = ({m for _, m in MODELS.get(a.model, [])} if modal else model_ids(a.model) - {a.model}) or {resolve(a.model)[1]}
		past = [r for f in flows(env) for rows in load_variants(f).values() for r in rows if r.get('status') == 'ok' and r.get('model') in mids | {f'modal:{m}' for m in mids}]
		if not past:
			return print(f'{n} episodes on {a.model}: not measured on this env yet - run a pilot first (--limit 5) and dry-run again')
		usd, sec = sum(r['cost_usd'] for r in past) / len(past), sum(r['latency_s'] for r in past) / len(past)
		return print(
			f'{n} episodes on {a.model}: ~${n * usd:.2f}, ~{n * sec / a.concurrency / 60:.0f} min at concurrency {a.concurrency} (measured on {len(past)} earlier episodes)'
		)
	if a.variant is None:  # a comparison flow keeps one model per slot, named in change.md (the report's column title)
		flow = (a.compare_flow or 'hillclimb') if a.blind else (a.flow or 'hillclimb')  # blind: the slot this model has where it ran with materials
		a.variant = 'baseline' if flow == 'hillclimb' else model_slot(env / flow, a.model)
	if not re.fullmatch(r'baseline|v\d+', a.variant):
		raise SystemExit('variant must be `baseline` or v<N> (the report reads only these; the model name goes in change.md)')
	from . import pipeline

	v = run(pipeline.run(env, spec_of(env), a))
	print(f'-> {v}/results.jsonl')
	if a.blind:  # what the materials are worth: the same model with and without them, on the same cases
		blind = {r['prompt_id']: r['grade']['correct'] for r in load_jsonl(v / 'results.jsonl') if r.get('status') == 'ok'}
		seen = {
			r['prompt_id']: r['grade']['correct']
			for r in load_jsonl(env / (a.compare_flow or 'hillclimb') / a.variant / 'results.jsonl')
			if r.get('status') == 'ok' and r['prompt_id'] in blind
		}
		both = [p for p in blind if p in seen]
		if both:
			print(f'{a.model} on {len(both)} cases: {sum(seen[p] for p in both) / len(both):.0%} with the materials, {sum(blind[p] for p in both) / len(both):.0%} without')
		return
	report(env, judges=True)


def model_slot(flow: Path, model: str) -> str:
	"""The slot of `model` in a comparison flow: the one whose change.md names it, else the next free one."""
	from .llm import model_ids

	ids = model_ids(model)
	named = lambda d: (d / 'change.md').read_text().strip() == model if (d / 'change.md').exists() else any(r.get('model') in ids for r in load_jsonl(d / 'results.jsonl')[:1])  # noqa: E731
	slot = next((d.name for d in sorted(flow.glob('*')) if d.is_dir() and named(d)), None) if flow.exists() else None
	if slot is None:
		taken = {d.name for d in flow.iterdir() if d.is_dir()} if flow.exists() else set()
		slot = next(v for v in ['baseline'] + [f'v{i}' for i in range(1, 1000)] if v not in taken)
		(flow / slot).mkdir(parents=True, exist_ok=True)
		(flow / slot / 'change.md').write_text(model + '\n')
	return slot


def cmd_compare(a):
	from .pipeline import load_kept
	from .vanilla import compare_mining, solve

	env = Path(a.env)
	spec = spec_of(env)
	if a.missions:
		ms = json.loads((env / 'kb.json').read_text())['missions']
		by_src = {}
		for m in [m for m in ms if m.get('angle') != 'outcome' and 'error' not in m]:  # spread over sources: the n first missions per start host, round-robin
			by_src.setdefault(m['start_url'].split('/')[2], []).append({k: m[k] for k in ('angle', 'instructions', 'start_url')})
		pick = [x for i in range(a.missions) for v in by_src.values() if i < len(v) for x in [v[i]]][: a.missions]
		for arm, r in run(compare_mining(env, spec, pick, a.cc_model, a.agent_model)).items():
			print(
				f'{arm:12s} '
				+ (f'{r["complete_cases"]} cases -> {r["tasks_kept"]}/{r["tasks_drafted"]} tasks kept, ' if 'complete_cases' in r else '')
				+ f'{r["verified_facts"]:3d} verified facts ({r["browser_only"]} browser-only) from {len(r["sites"])} sites, mining ${r["usd"]}, {r["minutes"]} min  {r["sites"]}'
			)
	if a.variant:
		print(f'-> {run(solve(load_kept(env), env / "hillclimb", a.variant, a.cc_model, a.timeout, a.concurrency))}/results.jsonl')
		report(env, judges=False)


def cmd_stop(a):
	from .cloud import stop_all

	ids = stop_all()
	print(f'stopped {len(ids)} rl-env-use Modal app(s) and their containers' + (f': {" ".join(ids)}' if ids else ' - none was running'))


def cmd_report(a):
	report(Path(a.env), judges=not a.no_judges)


def report(env: Path, judges: bool = True) -> None:
	"""Regrade every stored run with the current graders (+ trusted judges), audit, and write report.html."""
	from .audit import audit
	from .judges import load_judges

	trusted = load_judges(env / 'judges.json', only_trusted=True)
	if judges and trusted:
		run(apply_judges(env, trusted))
	regrade(env, {j.id for j in load_judges(env / 'judges.json')})
	for f in flows(env):
		write_state(env, trusted, f.name)
	findings = audit(env)
	(env / 'audit.json').write_text(json.dumps(findings, indent=1))
	for f in findings:
		print(f'[{f["severity"]:6s}] {f["check"]:14s} {f["message"]}')
	for f in flows(env):
		try:
			subprocess.run(['node', str(REPORT_BUILDER), str(f)], check=True)
		except (FileNotFoundError, subprocess.CalledProcessError) as e:
			print(f'{f.name}/report.html not written ({e}); install Node.js to get the HTML report')


async def apply_judges(env: Path, judges) -> None:
	"""Calibrated narrow judges on every stored conversation that has no verdict yet (verdicts are cached in the row)."""
	from .audit import load_variants

	cases = {c['id']: c for c in load_jsonl(env / 'tasks.jsonl')}
	sem = asyncio.Semaphore(4)
	for flow, v, rows in [(f, v, rows) for f in flows(env) for v, rows in load_variants(f).items()]:

		async def one(r, flow=flow, v=v):
			tp = flow / v / 'traces' / f'{r["prompt_id"]}_rep{r["rep"]}.json'
			for j in judges:
				if j.id not in r['grade'] and r['prompt_id'] in cases and tp.exists():
					async with sem:
						res = await j.run(cases[r['prompt_id']], json.loads(tp.read_text()), r['meta'].get('answer'))
					r['grade'][j.id] = res['score']
					r.setdefault('explanation', {})[j.id] = res['reasoning']

		await asyncio.gather(*(one(r) for r in rows if r.get('status') == 'ok'))
		save_jsonl(flow / v / 'results.jsonl', rows)


def regrade(env: Path, judge_ids: set) -> None:
	"""Re-score stored outputs with the current graders and golds; rows of cases removed since are set aside, not lost."""
	from .audit import load_variants
	from .pipeline import regrade_row
	from .tasks import keep, load_tasks

	cases = {
		c['id']: c for c in load_tasks(env) if keep(c, verified=True)
	}  # verified, not only panel-hard (pool runs score easy tasks too), and before outcome balancing: a forecast held back for balance keeps its rows
	for flow, v, rows in [(f, v, rows) for f in flows(env) for v, rows in load_variants(f).items()]:
		dp = flow / v / 'dropped.jsonl'
		old = load_jsonl(dp)  # cases re-admitted since get their rows back
		done = {(r['prompt_id'], r['rep']) for r in rows}
		rows += [r for r in old if r['prompt_id'] in cases and (r['prompt_id'], r['rep']) not in done]
		dropped = [r for r in rows if r['prompt_id'] not in cases]
		rows = [r for r in rows if r['prompt_id'] in cases]
		for r in rows:
			if r.get('status') == 'ok':
				r['grade'] = {k: x for k, x in r['grade'].items() if k in judge_ids} | regrade_row(flow / v, cases[r['prompt_id']], r)  # judge verdicts are cached
		save_jsonl(flow / v / 'results.jsonl', rows)
		if dropped or old or dp.exists():  # written after results: crash-safe
			save_jsonl(dp, [r for r in old if r['prompt_id'] not in cases] + dropped)
	sp = env / 'hillclimb' / '_state.json'
	if sp.exists():
		s = json.loads(sp.read_text())
		for k in ('train_ids', 'test_ids'):
			if k in s:
				s[k] = [i for i in s[k] if i in cases]
		sp.write_text(json.dumps(s, indent=1))


def forget_verdicts(env: Path, judge_id: str) -> None:
	from .audit import load_variants

	for flow in flows(env):
		for v, rows in load_variants(flow).items():
			for r in rows:
				r['grade'].pop(judge_id, None)
				r.get('explanation', {}).pop(judge_id, None)
			save_jsonl(flow / v / 'results.jsonl', rows)


def labels(env: Path, annotator: str) -> list[dict]:
	mine = [r for r in load_jsonl(env / 'annotations.jsonl') if annotator == 'any' or r.get('annotator', 'human') == annotator]
	return list({r['id']: r for r in mine}.values())  # filter by annotator BEFORE taking the latest label


def cmd_codify(a):
	from .judges import codify, mode_counts

	env = Path(a.env)
	ann = labels(env, a.annotator)
	tax = run(codify(ann, a.model))
	counts = mode_counts(tax, ann)
	n_fail = sum(x['verdict'] == 'fail' for x in ann)
	(env / 'failure_modes.json').write_text(
		json.dumps({'modes': [m.model_dump() | {'count': counts[m.id]} for m in tax.modes], 'n_annotated': len(ann), 'n_failed': n_fail}, indent=1)
	)
	print(f'{len(ann)} annotated, {n_fail} failed. Failure modes by frequency (build evaluators for the top ones only):')
	for m in sorted(tax.modes, key=lambda m: -counts[m.id]):
		print(f'  {counts[m.id]:3d}  {m.id:28s} {m.definition}')


def cmd_judges(a):
	from .judges import FailureMode, Judge, calibrate, load_judges, save_judges

	env = Path(a.env)
	modes = {m['id']: FailureMode(**{k: v for k, v in m.items() if k != 'count'}) for m in json.loads((env / 'failure_modes.json').read_text())['modes']}
	cases = {c['id']: c for c in load_jsonl(env / 'tasks.jsonl')}
	labeled = []
	for iid, lab in {r['id']: r for r in labels(env, a.annotator)}.items():
		variant, rest = iid.split('/', 1)
		pid, rep = rest.rsplit('_rep', 1)
		tp = env / 'hillclimb' / variant / 'traces' / f'{pid}_rep{rep}.json'
		rows = [r for r in load_jsonl(env / 'hillclimb' / variant / 'results.jsonl') if r['prompt_id'] == pid and r['rep'] == int(rep)]
		if tp.exists() and rows and pid in cases:
			labeled.append((cases[pid], json.loads(tp.read_text()), rows[0]['meta'].get('answer'), lab))
	judges = {j.id: j for j in load_judges(env / 'judges.json')}  # judges of other modes are kept
	todo = [Judge(modes[m], a.model) for m in (a.modes.split(',') if a.modes else modes)]

	async def calibrate_all():  # one event loop for every judge: HTTP clients must not outlive their loop
		for j in todo:
			data = [(t, tr, ans, j.id in lab.get('modes', [])) for t, tr, ans, lab in labeled]
			if not any(h for *_, h in data):
				print(f'  {j.id}: no traces tagged with this mode yet - tag some in review first')
				continue
			s = await calibrate(j, data) | {'annotator': a.annotator}
			j.stats = s
			print(f'  {j.id}: TPR {s["tpr"]} TNR {s["tnr"]} flip {s.get("flip_rate")} n={s["n"]} -> {"TRUSTED" if s["trusted"] else "NOT trusted"}', flush=True)
			judges[j.id] = j
			save_judges(env / 'judges.json', list(judges.values()))
			forget_verdicts(env, j.id)  # the recalibrated judge re-scores every run on the next report

	run(calibrate_all())


def cmd_split(a):
	from .audit import split

	env = Path(a.env)
	from .pipeline import load_kept

	train, test = split(load_kept(env), a.test_frac, a.seed)
	write_state(env)
	p = env / 'hillclimb' / '_state.json'
	p.write_text(json.dumps(json.loads(p.read_text()) | {'train_ids': train, 'test_ids': test, 'split_seed': a.seed}, indent=1))
	print(f'train {len(train)} / test {len(test)} (stratified, seed {a.seed}) -> {p}')


def cmd_export(a):
	from .pipeline import export

	env = Path(a.env)
	print(f'{export(env, env / "dataset.jsonl")} tasks -> {env}/dataset.jsonl')


def debug_signals() -> None:
	"""kill -USR1 <pid>: thread stacks; kill -USR2 <pid>: every pending asyncio task's stack (where a stalled run waits)."""
	import faulthandler
	import signal

	def tasks(*_):
		try:
			for t in asyncio.all_tasks(asyncio.get_event_loop()):
				t.print_stack(limit=12, file=sys.stderr)
		except RuntimeError as e:
			print(f'no event loop: {e}', file=sys.stderr)
		sys.stderr.flush()

	try:
		faulthandler.register(signal.SIGUSR1, all_threads=True)
		signal.signal(signal.SIGUSR2, tasks)
	except (ValueError, OSError, AttributeError):  # no real stderr (captured) or not the main thread
		pass


def main(argv=None):
	p = argparse.ArgumentParser(prog='rl-env-use', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sp = p.add_subparsers(dest='cmd', required=True, metavar='command')
	q = sp.add_parser('create', help='autonomous: one line -> a verified, calibrated env')
	q.add_argument(
		'line',
		help='the domain and skill in your words, with any sources or constraints, e.g. "credit analysis of small-cap 10-Qs: compute leverage and forecast covenant breaches; SEC EDGAR"',
	)
	q.add_argument('--name', help='env name (default: planned); an existing env of that name is resumed')
	q.add_argument('--tasks', type=int, help='target number of kept tasks (default 150; a resume keeps its own)')
	q.add_argument('--budget', type=float, help='$ cap for mining, writing, verifying and calibrating: API-equivalent model cost + Modal time (default 40; a resume keeps its own)')
	q.add_argument('--missions', type=int, default=12, help='mining missions in the first round (later rounds are sized from the measured yield)')
	q.add_argument('--model', default='sonnet46', help='planner, task writer and critic')
	q.add_argument('--panel', nargs='+', help='difficulty panel (default: the frontier, opus55 gpt-astra); a task is dropped only if every panel model solves it')
	q.add_argument('--mix', choices=['derive', 'predict', 'mixed'], help='override the planned task kinds')
	q.add_argument('--plan-only', action='store_true', help='print the planned spec and stop')
	q = sp.add_parser('climb', help="autonomous: hillclimb a policy's system prompt on an env")
	q.add_argument('env')
	q.add_argument('--model', required=True, help='the policy to climb (e.g. qwen-35b, qwen-27b, haiku, or modal:<hf-org>/<model>)')
	q.add_argument('--rounds', type=int, default=8)
	q.add_argument('--k', type=int, default=3, help='candidate edits per round (screened on train, the best goes to test)')
	q.add_argument('--reps', type=int, default=3)
	q.add_argument('--budget', type=float, default=20.0)
	q.add_argument('--proposer', default='sonnet46', help='model that reads train failures and proposes edits')
	q.add_argument('--system', help='starting system prompt extension (file)')
	q.add_argument('--test-frac', type=float, default=0.5)
	q.add_argument('--seed', type=int, default=0)
	q.add_argument('--confidence', type=float, default=0.95, help='keep an edit only if P(test gain > 0) under the case bootstrap reaches this')
	q.add_argument('--mde', type=float, default=0.08, help='refuse to climb when the smallest detectable gain exceeds this (use --force to climb anyway)')
	q.add_argument('--force', action='store_true')
	q.add_argument('--concurrency', type=int, default=8)
	q.add_argument('--verified', action='store_true', help="climb on every verified task, not only the panel-hard ones (a small policy's training tier)")
	q = sp.add_parser('new', help='create an env spec')
	q.add_argument('name')
	q.add_argument('--topic', required=True, help='the domain, in your words')
	q.add_argument(
		'--skill',
		help='what the model must do, input -> outcome, e.g. "given a 10-Q, predict whether the next 10-K carries a going-concern warning". Omit for browse tasks (the model finds answers on the live web)',
	)
	q.add_argument('--sources', nargs='+', default=[], help='where the data lives: portals, databases, search pages, forums')
	q.add_argument('--proxy', help='proxy for every browser of this env (Chromium --proxy-server, e.g. http://host:3128), for geo-restricted sources')
	q = sp.add_parser('build', help='mine the sources, then write and verify tasks')
	q.add_argument('env')
	q.add_argument('--only', help='(re)run one stage: scout (another mining round, appended), look, synth (more tasks, appended), verify, panel, harden')
	q.add_argument('--model', default='sonnet46', help='model for planning, task writing and checks')
	q.add_argument('--enrich', action='store_true', help='with --only scout: one mission per complete case to save the data its computations need')
	q = sp.add_parser('review', help='local app for the mined data, tasks and traces')
	q.add_argument('env')
	q.add_argument('--variant', default='baseline')
	q.add_argument('--port', type=int, default=8765)
	q = sp.add_parser('run', help='run the policy under test, grade, audit, report')
	q.add_argument('env')
	q.add_argument('--model', help='policy model: opus55, gpt-astra, sonnet46, haiku, qwen-35b, qwen-27b, modal:<hf-org>/<model>, ... (default sonnet46)')
	q.add_argument('--variant', help='baseline or v<N> (default: baseline; in a comparison --flow, the slot of --model)')
	q.add_argument('--reps', type=int, default=1)
	q.add_argument('--timeout', type=float, default=600)
	q.add_argument('--concurrency', type=int, default=4)
	q.add_argument('--limit', type=int)
	q.add_argument('--system', help='system prompt file for the policy - the surface to hillclimb')
	q.add_argument('--dry-run', action='store_true', help='print episodes, estimated cost and time, then stop')
	q.add_argument('--flow', help="flow directory under the env (default hillclimb): keep model comparisons (e.g. `models`) out of the climb's rounds")
	q.add_argument('--blind', action='store_true', help='ablation: withhold the materials (written to ablation/<variant>, compared with <--compare-flow>/<variant>)')
	q.add_argument('--compare-flow', help='with --blind: the flow holding the same variant run with materials (default hillclimb)')
	q.add_argument('--verified', action='store_true', help='run every verified task, also those the panel found easy (model comparisons; training a small policy)')
	q = sp.add_parser('compare', help='A/B against vanilla Claude Code browsing (WebSearch + WebFetch)')
	q.add_argument('env')
	q.add_argument('--missions', type=int, default=0, help="mining A/B: rerun N of the env's missions with both harnesses")
	q.add_argument('--variant', help="solving A/B: run Claude Code on the env's tasks as this hillclimb variant (v<N>)")
	q.add_argument('--cc-model', default='claude-sonnet-4-6', help='model for the Claude Code arm')
	q.add_argument('--agent-model', default='sonnet46', help='the same model as a browser agent (browser-harness-js on Modal) for the mining arm')
	q.add_argument('--timeout', type=float, default=900)
	q.add_argument('--concurrency', type=int, default=4)
	sp.add_parser('stop', help='stop every Modal container of rl-env-use still running in your workspace')
	q = sp.add_parser('report', help='regrade, audit, write report.html')
	q.add_argument('env')
	q.add_argument('--no-judges', action='store_true', help='skip applying calibrated judges to new runs')
	for name, h in (
		('codify', 'failure taxonomy from your trace notes'),
		('judges', 'calibrate narrow judges on your labels'),
		('split', 'train/test split'),
		('export', 'JSONL dataset'),
	):
		q = sp.add_parser(name, help=h)
		q.add_argument('env')
		if name in ('codify', 'judges'):
			q.add_argument('--model', default='sonnet46')
			q.add_argument('--annotator', default='human', help="whose labels to use: 'human', a subagent name, or 'any'")
		if name == 'judges':
			q.add_argument('--modes', help='comma-separated failure mode ids (default: all)')
		if name == 'split':
			q.add_argument('--test-frac', type=float, default=0.5)
			q.add_argument('--seed', type=int, default=0)
	a = p.parse_args(argv)
	sys.stdout.reconfigure(line_buffering=True)  # progress lines reach a redirected log as they happen
	debug_signals()
	import signal

	for sig in (signal.SIGINT, signal.SIGTERM):  # a kill cleans up like Ctrl-C (a job started with & ignores SIGINT otherwise)
		signal.signal(sig, signal.default_int_handler)
	load_dotenv()
	try:
		globals()[f'cmd_{a.cmd}'](a)
	finally:
		if LEDGER.by_model:
			print(f'spend: ${LEDGER.usd:.3f}  {json.dumps(LEDGER.by_model)}')
		if a.cmd in ('build', 'run', 'compare', 'create', 'climb') and not getattr(
			a, 'dry_run', False
		):  # the Modal client and CLI pipes leave non-daemon threads alive (containers are already stopped)
			if sys.exc_info()[0]:  # os._exit skips the interpreter's traceback printing
				import traceback

				traceback.print_exc()
			sys.stdout.flush()
			sys.stderr.flush()
			os._exit(1 if sys.exc_info()[0] else 0)


if __name__ == '__main__':
	sys.exit(main())
