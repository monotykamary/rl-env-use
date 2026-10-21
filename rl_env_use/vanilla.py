"""The control arm: the vanilla Claude setup the claude-api skill runs in (Claude Code with WebSearch + WebFetch), on the
same missions and tasks as the browser agents (agent.py on browser-harness-js). Same model, same instructions, same
checks - only the browsing differs, so the gap between the arms is the measured edge."""

import asyncio
import json
import re
import time
from pathlib import Path

from .browse import SEARCH_ENGINES, browser, capture, quote_in
from .graders import grade_answer, norm
from .llm import CLAUDE_BARE, LEDGER, now
from .rollout import trace_grade
from .runner import run_eval
from .scout import fact_issue, slice_passage


async def claude_web(prompt: str, model: str = 'claude-sonnet-4-6', timeout: float = 900) -> tuple[str, list[dict], float]:
	"""One Claude Code run with only WebSearch/WebFetch. Returns (final text, trace in our format, API-equivalent $)."""
	args = ['claude', '-p', '--model', model, '--output-format', 'stream-json', '--verbose', '--tools', 'WebSearch,WebFetch', '--allowedTools', 'WebSearch,WebFetch', *CLAUDE_BARE]
	proc = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd='/tmp')
	try:
		out, _ = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout)
	except (TimeoutError, asyncio.CancelledError):
		proc.kill()
		raise
	trace, result, usd, names, step, failed = [{'role': 'system', 'content': prompt}], '', 0.0, {}, 0, True  # no result event = failed
	for line in out.decode().splitlines():
		d = json.loads(line) if line.startswith('{') else {}
		content = d.get('message', {}).get('content')
		for b in content if isinstance(content, list) else []:
			if b.get('type') == 'tool_use' and b['name'] in ('WebFetch', 'WebSearch'):
				names[b['id']] = 'extract' if b['name'] == 'WebFetch' else 'search'  # a fetched page counts as read, like a browser extract
				url = b['input'].get('url') or f'https://www.google.com/search?q={b["input"].get("query", "")}'
				trace += [{'role': 'assistant', 'content': f'[step {step} @ {url}]'}, {'role': 'tool_call', 'name': names[b['id']], 'content': json.dumps(b['input'])}]
				step += 1
			elif b.get('type') == 'tool_result' and b.get('tool_use_id') in names:
				c = b.get('content')
				trace.append({'role': 'tool_result', 'content': c if isinstance(c, str) else json.dumps(c)[:20000]})
		if d.get('type') == 'result':
			result, usd, failed = d.get('result') or '', d.get('total_cost_usd') or 0.0, d.get('is_error')
	LEDGER.add(f'claude-code/{model}', {'reported_cost': usd})
	if proc.returncode or failed:  # a quota, auth or rate-limit stop is an infra error, retried on resume - never a wrong answer
		raise RuntimeError(f'claude -p failed (exit {proc.returncode}): {result[:200]}')
	return result, trace, usd


def last_json(text: str) -> dict:
	dec, i, last = json.JSONDecoder(), 0, {}
	while (i := text.find('{', i)) >= 0:  # left to right, skipping over each parsed object: the last top-level one wins
		try:
			obj, end = dec.raw_decode(text, i)
			last, i = obj if isinstance(obj, dict) else last, end
		except ValueError:
			i += 1
	return last


async def verify_facts(facts: list[dict], kb: dict, angle: str, miner: str, materials: list[dict] = ()) -> int:
	"""Arm-neutral evidence bar: the quote must be on the live page as a person sees it (our browser, expanded) and
	pass the same fact rules as the browser miner's record_fact. Materials are cut from that page by the same
	markers rule as record_material - we do the cutting, so this arm only has to point at the passage."""
	session, pages, n = browser(), {}, 0
	try:
		await session.start()
		for m in materials:
			url = str(m.get('url', ''))
			if not url.startswith('http'):
				continue
			if url not in pages:
				try:
					pages[url] = (await capture(session, url))['text']
				except Exception:
					pages[url] = ''
			p = slice_passage(pages[url], str(m.get('starts_with', '')), str(m.get('ends_with', '')))
			if not p.startswith('!'):
				kb.setdefault('materials', []).append(
					{'id': f'm{len(kb.get("materials", []))}', 'case': str(m.get('case', '')), 'title': str(m.get('title', '')), 'text': p, 'url': url, 'miner': miner, 'at': now()}
				)
		for f in facts:
			url, quote, value = str(f.get('url', '')), str(f.get('quote', '')), str(f.get('value', ''))
			if not url.startswith('http') or SEARCH_ENGINES.match(url) or fact_issue(value, quote):
				continue
			if url not in pages:
				try:
					pages[url] = (await capture(session, url))['text']
				except Exception:
					pages[url] = ''
			if quote_in(quote, pages[url]) and not any(x['url'] == url and norm(x['value']) == norm(value) for x in kb['facts']):
				kb['facts'].append(
					{
						'id': f'f{len(kb["facts"])}',
						'claim': str(f.get('claim', '')),
						'value': value,
						'quote': quote,
						'url': url,
						'angle': angle,
						'miner': miner,
						'at': now(),
						**({'case': str(f['case'])} if f.get('case') else {}),
					}
				)
				n += 1
	finally:
		await session.kill()
	return n


MINE = """You are scouting the topic "{topic}" to build a benchmark of realistic web tasks.
Mission ({angle}): {instructions}
Start at {start_url}. Aim for 8-12 facts about several DIFFERENT entities from DIFFERENT pages, including some reachable only after several steps.
Facts must be short atomic values that will not change soon. Every quote must be copied verbatim from the page at `url` and contain the value plus at least 2 surrounding words that say what it means.
End your reply with one JSON object: {{"facts": [{{"claim": ..., "value": ..., "quote": ..., "url": ...}}], "observations": [what made this hard]}}"""


MINE_CASES = """You are collecting real cases for an environment that trains a model in "{topic}".
The model will be given the input materials of a case and must: {skill}
Mission: {instructions}
Start at {start_url}. Aim for 3-6 complete cases about DIFFERENT entities. For each case pick a short key, point at its input passage (the page url plus the exact first and last few words of a 300-20000 character passage a solver needs; it must not state the outcome), and give its true outcome as facts copied verbatim from the page that states it, with at least 2 surrounding words that say what the value means.
End your reply with one JSON object: {{"materials": [{{"case": ..., "title": ..., "url": ..., "starts_with": ..., "ends_with": ...}}], "facts": [{{"case": ..., "claim": ..., "value": ..., "quote": ..., "url": ...}}], "observations": [what made this hard]}}"""


async def mine(missions: list[dict], topic: str, model: str, concurrency: int = 4, skill: str | None = None) -> dict:
	"""The vanilla miner on the given missions; every returned fact and material goes through `verify_facts`."""
	kb, sem = {'topic': topic, 'facts': [], 'ideas': [], 'sources': [], 'missions': []}, asyncio.Semaphore(concurrency)

	async def one(m):
		async with sem:
			t0 = time.monotonic()
			try:
				text, trace, usd = await claude_web(MINE_CASES.format(topic=topic, skill=skill, **m) if skill else MINE.format(topic=topic, **m), model)
			except Exception as e:
				kb['missions'].append({**m, 'error': f'{type(e).__name__}: {e}'[:300]})
				return
			out = last_json(text)
			facts, mats = out.get('facts', []), out.get('materials', [])
			kb['missions'].append(
				{
					**m,
					'raw_facts': facts,
					'raw_materials': mats,
					'returned': len(facts),
					'verified': await verify_facts(facts, kb, m['angle'], 'claude-code', mats),
					'fetches': sum(t['role'] == 'tool_call' for t in trace),
					'usd': usd,
					'latency_s': round(time.monotonic() - t0),
				}
			)

	await asyncio.gather(*(one(m) for m in missions))
	return kb


SOLVE = '{prompt}\n\nUse the web. End with two lines exactly:\nANSWER: <just the answer: value, name, date or comma-separated list>\nSOURCE: <url of the page you read it on>'


async def solve(tasks: list[dict], flow: Path, variant: str, model: str, timeout_s: float = 900, concurrency: int = 4) -> Path:
	"""The vanilla policy on the env's tasks, written as one more hillclimb variant (same graders, same report)."""

	async def one(t, rep, vdir):
		t0 = time.monotonic()
		text, trace, usd = await claude_web(SOLVE.format(prompt=t['prompt']), model, timeout_s)
		m = re.findall(r'^\W*ANSWER:\s*(.+)$', text, re.M)
		answer = m[-1].strip().strip('*').strip() if m else None
		correct = grade_answer(t, answer)
		trace.append({'role': 'assistant', 'content': f'FINAL ANSWER: {answer}'})
		return trace, {
			'prompt_id': t['id'],
			'rep': rep,
			'prompt': t['prompt'],
			'tags': [t.get('category', 'task'), t.get('difficulty', '?')],
			'grade': trace_grade(t, correct, trace),
			'status': 'ok',
			'stop_reason': 'done',
			'model': f'claude-code/{model} (WebSearch+WebFetch)',
			'latency_s': round(time.monotonic() - t0, 2),
			'tool_calls': sum(x['role'] == 'tool_call' for x in trace),
			'steps': sum(x['role'] == 'tool_call' for x in trace),
			'usage': {'reported_cost': usd},
			'cost_usd': round(usd, 5),
			'meta': {
				'answer': answer,
				'gold': t['answer'],
				'failure_class': None if correct else ('no_answer' if answer is None else 'wrong_answer'),
				'agent_claimed_success': answer is not None,
				'at': now(),
			},
		}

	return await run_eval(tasks, one, flow, variant, 1, timeout_s + 60, concurrency)


async def compare_mining(env: Path, spec: dict, missions: list[dict], cc_model: str = 'claude-sonnet-4-6', agent_model: str = 'sonnet46') -> dict:
	"""Same missions, same model, two harnesses. Browser-agent facts are verified on the live page as they are recorded,
	Claude Code facts afterwards (on the expanded page - the more lenient check). Then: what would plain HTTP have seen."""
	from .pipeline import look
	from .scout import Mission, run_mission

	out = {}
	for arm in ('claude_code', 'browser_agent'):
		d = env / 'compare' / arm
		d.mkdir(parents=True, exist_ok=True)
		usd0, t0 = LEDGER.usd, time.monotonic()
		if arm == 'claude_code':
			kb = await mine(missions, spec['topic'], cc_model, spec.get('concurrency', 4), spec.get('skill'))
		else:
			kb, sem = {'topic': spec['topic'], 'facts': [], 'ideas': [], 'sources': [], 'missions': []}, asyncio.Semaphore(spec.get('concurrency', 4))

			async def one(m, kb=kb, sem=sem):
				async with sem:
					try:
						kb['missions'].append(
							await run_mission(
								Mission(**{k: m[k] for k in ('angle', 'instructions', 'start_url')}),
								spec['topic'],
								kb,
								agent_model,
								spec.get('max_steps', 40),
								True,
								skill=spec.get('skill'),
							)
						)
					except (Exception, asyncio.CancelledError) as e:
						kb['missions'].append({**m, 'error': f'{type(e).__name__}: {e}'[:300]})

			await asyncio.gather(*(one(m) for m in missions))
		(d / 'kb.json').write_text(json.dumps(kb, indent=1))
		summary = await look(d)
		facts = json.loads((d / 'kb.json').read_text())['facts']
		mining_usd = round(LEDGER.usd - usd0, 3)
		case_stats = {}
		if spec.get('skill'):  # the product is tasks, not facts: each arm's data goes through the same synth + verify
			from .pipeline import build
			from .store import load_jsonl
			from .tasks import cases, keep

			for stage in ('synth', 'verify'):
				await build(d, spec, stage)
			tasks, kb = load_jsonl(d / 'tasks.jsonl'), json.loads((d / 'kb.json').read_text())
			case_stats = {'complete_cases': len(cases(kb)), 'materials': len(kb.get('materials', [])), 'tasks_drafted': len(tasks), 'tasks_kept': sum(keep(t) for t in tasks)}
		out[arm] = {
			'missions': len(missions),
			'verified_facts': len(facts),
			'returned': sum(m.get('returned', 0) for m in kb['missions']) if arm == 'claude_code' else None,
			'browser_only': sum(f.get('http') not in ('visible', None) for f in facts),
			'sites': sorted({f['url'].split('/')[2] for f in facts}),
			'usd': mining_usd,
			'minutes': round((time.monotonic() - t0) / 60, 1),
			'look': summary,
			**case_stats,
		}
	(env / 'compare' / 'mining.json').write_text(json.dumps(out, indent=1))
	return out
