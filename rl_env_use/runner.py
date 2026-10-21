"""Generic eval runner: any (case -> trace, row) function, written in the claude-api hillclimb contract.

<flow>/<variant>/results.jsonl     one row per (case, rep), appended as each completes (resume-safe at the (case, rep) key)
<flow>/<variant>/traces/<id>_rep<k>.json
<flow>/<variant>/errors.jsonl      infra failures (timeouts, crashes) - never scored as model failures
"""

import asyncio
import json
import re
from pathlib import Path

from . import ui
from .llm import now
from .store import load_jsonl

REFUSAL = re.compile(r'safeguards flagged|stop_reason.{0,5}refusal|declined to respond', re.I)  # a policy decline: neither a capability miss nor an infra fault


def healed(path: Path, f) -> None:
	"""A killed run can leave an unterminated last line: start the next row on its own line, never fused to it."""
	if path.stat().st_size and path.read_bytes()[-1:] != b'\n':
		f.write('\n')


def done_keys(path: Path) -> set:
	if not path.exists():
		return set()
	return {(r['prompt_id'], r['rep']) for r in load_jsonl(path) if r.get('status') in ('ok', 'truncated')}  # a cut last line is run again


async def run_eval(cases: list[dict], run_one, flow: Path, variant: str = 'baseline', reps: int = 1, timeout_s: float = 600, concurrency: int = 4, first: int = 0) -> Path:
	"""run_one(case, rep, vdir) -> (trace, row). Hard wall-clock ceiling per case; infra failures go to errors.jsonl."""
	vdir = flow / variant
	(vdir / 'traces').mkdir(parents=True, exist_ok=True)
	res, err = vdir / 'results.jsonl', vdir / 'errors.jsonl'
	skip = done_keys(res)
	sem = asyncio.Semaphore(concurrency)
	lock = asyncio.Lock()

	async def one(c, rep):
		for attempt in range(3):
			async with sem:
				try:
					trace, row = await asyncio.wait_for(run_one(c, rep, vdir), timeout_s)
					break
				except (Exception, asyncio.CancelledError) as e:  # a cancelled inner wait surfaces as CancelledError
					if isinstance(e, asyncio.CancelledError) and asyncio.current_task().cancelling():
						raise  # a real outer cancellation (Ctrl-C) must propagate
					if not (('Too many concurrent' in str(e) or 'HTTP 429' in str(e)) and attempt < 2):  # rate-limited before it started: retry
						cls = 'timeout' if isinstance(e, asyncio.TimeoutError) else 'refusal' if REFUSAL.search(str(e)) else 'harness_error'
						async with lock:
							with err.open('a') as f:
								healed(err, f)
								f.write(json.dumps({'prompt_id': c['id'], 'rep': rep, 'failure_class': cls, 'error': f'{type(e).__name__}: {e}'[:500], 'at': now()}) + '\n')
						return
			await asyncio.sleep(30)
		async with lock:
			(vdir / 'traces' / f'{c["id"]}_rep{rep}.json').write_text(json.dumps(trace, indent=1))
			with res.open('a') as f:
				healed(res, f)
				f.write(json.dumps(row) + '\n')

	await ui.gather((one(c, r) for c in cases for r in range(first, reps) if (c['id'], r) not in skip), f'run {variant}')
	return vdir
