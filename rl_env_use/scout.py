"""Stage 1: browser agents mine the sources and record evidence-backed facts, case materials, sources and task ideas.

Facts are only accepted if their quote is literally present in the live page text (checked via CDP at record time),
and materials are sliced from the live page text itself, so the knowledge base cannot contain hallucinated evidence.
A case = materials (what the model under test is given) + facts with the same case key (its true outcome).
"""

import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from . import ui
from .agent import Tool, ToolError, run_agent, tool
from .browse import SEARCH_ENGINES, Browser, browser, http_capture, quote_in
from .graders import norm
from .llm import LEDGER, call, now
from .store import write_atomic

ANGLES = {
	'database': "Mine a small table: pick ONE record type and 2-4 fixed fields (e.g. a score, a count, a position, a classification), then use the site's search, filters and pagination to open 5+ different records and record the SAME fields from each record page (prefer fields pinned to a versioned release over live counts that grow) - values only reachable through search or several clicks, never from the landing page.",
	'reference': 'Find the authoritative primary sources for this topic (official sites, registries, databases, documentation). Record facts that require navigating INTO these sites (detail pages, tables, filters), not facts from the landing page.',
	'longtail': 'Find specific, obscure, verifiable values (numbers, dates, names, lists) buried deep in pages or tables, that a person would only know by actually looking them up. Prefer stable facts that will not change week to week.',
	'users': 'Find what real people ask or struggle with on this topic (forums, Q&A sites, issue trackers, FAQ pages). Record the concrete questions people actually ask and, where the page answers them, the answer as a fact.',
	'deep': 'Find facts buried deep inside long documents: specific rows/cells of results tables and appendices in papers (prefer arXiv HTML versions), per-version entries in changelogs/release notes, specific comments inside closed issues. Record the table/section context in the quote. Skip anything stated in an abstract or README headline.',
	'cases': 'Collect complete CASES for the skill below, each about a DIFFERENT entity: open the page holding the input (a filing, a problem, a protocol, a case record) and save it with record_material; then find where the TRUE OUTCOME for that same entity is published (often a later or separate page: results, verdict, official solution, next report) and record it under the same case key: record_fact for one value, record_outcome for a results section or table (verbatim). A case without its outcome is worthless. The materials must not already state the outcome. If a document opens inside a heavy viewer, open its plain document version instead.',
	'interaction': 'Find tasks on this topic that need real site interaction: search forms, filters, sorting, pagination, comparing multiple pages, or reading a specific version/date of something. Record facts that can only be reached through such interaction.',
}


class Mission(BaseModel):
	angle: str
	instructions: str
	start_url: str = Field(description='Concrete https URL to start from (no search engines).')


class Plan(BaseModel):
	missions: list[Mission]


class ScoutSummary(BaseModel):
	observations: list[str] = Field(description='What makes this domain hard for a browser agent: navigation traps, blockers, ambiguity, data that changes.')


_RELATIVE = re.compile(r'\b(ago|today|yesterday|tomorrow|this (week|month|year)|last (week|month|year)|currently|now)\b', re.I)


def fact_issue(value: str, quote: str) -> str | None:
	"""The evidence bar every miner's facts must pass, whoever found them."""
	if len(norm(quote).split()) < len(norm(value).split()) + 2:  # table labels are terse: 'Enrollment (Actual) 28'
		return 'Quote must include the surrounding words that say what the value means (at least 2 words besides the value).'
	if len(value) > 80 or _RELATIVE.search(value):
		return 'Record a short, atomic, time-stable value (a number, date, name or short list) - not a sentence or a relative time like "3 days ago".'
	if not quote_in(value, quote):
		return 'The quote must contain the value.'
	if re.match(r'\s*[-\u2212\u2013(]', value) and not re.search(r'[-\u2212\u2013(]', quote):  # norm() drops signs: a loss must be one on the page
		return 'The value is negative but the quote shows no minus sign or parentheses: copy the value with the sign the page uses.'
	if re.match(r'\s*\$?\d', value) and re.search(r'(?<![\w\d])[-\u2212]\s?\$?\s?' + re.escape(value.strip().lstrip('$')), quote):
		return 'The quote shows this value with a minus sign: record the value with its sign.'
	return None


MATERIAL_CHARS = (300, 20000)
MISSION_TIMEOUT = 1500  # a browser stuck reconnecting once held a mission for 80 minutes; what it saved stays saved
MAX_MATERIALS = 3  # per case before its outcome is recorded: input is cheap, the outcome is the hard part


def marker(words: str) -> re.Pattern | None:
	"""A marker as page text renders it: any case (CSS uppercases headings), any punctuation or whitespace between words (table cells, quote styles)."""
	toks = re.findall(r'\w+', words)  # words only: table cells render with tabs, colons, pipes or nothing between them
	return re.compile(r'[\W_]*'.join(map(re.escape, toks)) + r'[^\w\s]*', re.I) if toks else None  # a closing '.' or ')' stays in the passage


def near(text: str, words: str) -> str:
	"""The page text where the first two words of a failed marker occur: lets the agent copy the real wording."""
	m = marker(' '.join(words.split()[:2]))
	hits = [h.start() for h in m.finditer(text)][:3] if m else []
	return (' Text on the page there: ' + ' | '.join(repr(' '.join(text[max(0, h - 80) : h + 100].split())) for h in hits)) if hits else ''


def slice_passage(text: str, starts_with: str, ends_with: str) -> str:
	"""The page text from the first `starts_with` through the next `ends_with`, verbatim; '!reason' when it cannot be cut."""
	s, e = marker(starts_with), marker(ends_with)
	i = s.search(text) if s else None
	if not i:
		return '!starts_with not found on the current page: copy the first words of the passage exactly as shown.' + near(text, starts_with)
	j = e.search(text, i.end()) if e else None
	if not j:
		return '!ends_with not found after starts_with: copy the last words of the passage exactly as shown.' + near(text[i.end() :], ends_with)
	passage = text[i.start() : j.end()]
	if not MATERIAL_CHARS[0] <= len(passage) <= MATERIAL_CHARS[1]:
		return f'!passage is {len(passage)} chars; it must be {MATERIAL_CHARS[0]}-{MATERIAL_CHARS[1]}: pick the part a solver needs.'
	return passage


def make_tools(kb: dict, angle: str, b: Browser, save=lambda: None, tally: dict | None = None) -> list[Tool]:
	tally = {} if tally is None else tally  # what this mission saved: the source-health record
	own = set()  # case keys this mission opened

	@tool(
		'Record one verifiable fact you can see on the CURRENT page. `quote` must be copied verbatim from the page and contain the value. `case`: the case key when this fact is the outcome of a case, else empty.'
	)
	async def record_fact(claim: str, quote: str, value: str = '', case: str = '') -> str:
		keys = {m['case'] for m in kb.get('materials', [])}
		if case and keys and case not in keys:  # an outcome must land on the case its material was saved under
			raise ToolError(f'Unknown case key {case!r}. Use the exact key you saved its material under: {sorted(keys)[-12:]}')
		if not value.strip():  # agent models often drop this field: a quote holding exactly one number names it
			nums = re.findall(r'[-+]?\$?\d[\d,]*(?:\.\d+)?%?', quote)
			if len(nums) != 1:
				raise ToolError('Give `value`: the short atomic value (number, date, name) inside the quote that the claim is about.')
			value = nums[0]
		if issue := fact_issue(value, quote):
			raise ToolError(issue)
		url, text = await b.url(), await b.text()
		if SEARCH_ENGINES.match(url):
			raise ToolError('Search result snippets are not evidence. Open the primary source page and record the fact there.')
		if not quote_in(quote, text):
			raise ToolError('Quote not found verbatim on the current page. Copy the exact text that contains the value.' + near(text, value))
		if any(f['url'] == url and norm(f['value']) == norm(value) and f.get('case') == (case or None) for f in kb['facts']):  # another case may share the value
			return f'Already recorded: {value}'
		kb['facts'].append(
			{'id': f'f{len(kb["facts"])}', 'claim': claim, 'value': value, 'quote': quote, 'url': url, 'angle': angle, 'at': now(), **({'case': case} if case else {})}
		)
		tally['outcomes' if case else 'facts'] = tally.get('outcomes' if case else 'facts', 0) + 1
		save()
		return f'Recorded fact #{len(kb["facts"])}: {claim} = {value}'

	@tool(
		'Save the OUTCOME of a case as a verbatim passage of the CURRENT page - a results section, a results table, a decision - when it is a table or several numbers (record_fact is for one value). Give the first and last few words exactly as on the page; everything between is saved. `case`: the key its material was saved under. Save what the page shows; never a computed or summarized value.'
	)
	async def record_outcome(case: str, title: str, starts_with: str, ends_with: str) -> str:
		keys = {m['case'] for m in kb.get('materials', [])}
		if case not in keys:
			raise ToolError(f"Unknown case key {case!r}: save the case's material first, or use the exact key it was saved under: {sorted(keys)[-12:]}")
		url, text = await b.url(), await b.text()
		if SEARCH_ENGINES.match(url):
			raise ToolError('Search result snippets are not evidence. Open the primary source page.')
		passage = slice_passage(text, starts_with, ends_with)
		if passage.startswith('!'):
			raise ToolError(passage[1:])
		if any(f.get('case') == case and passage in f['quote'] for f in kb['facts']):
			return f'Already saved for case {case}.'
		kb['facts'].append(
			{'id': f'f{len(kb["facts"])}', 'claim': title, 'value': '(results passage)', 'quote': passage, 'url': url, 'angle': angle, 'at': now(), 'case': case, 'passage': True}
		)
		tally['outcomes'] = tally.get('outcomes', 0) + 1
		save()
		return f'Saved outcome for case {case}: {title} ({len(passage)} chars)'

	@tool(
		'Save a passage of the CURRENT page as input material for a case (what the model under test will read). Give the first and last few words of the passage exactly as on the page; everything between them is saved verbatim. `case`: a short key naming the entity, reused for its outcome fact.'
	)
	async def record_material(case: str, title: str, starts_with: str, ends_with: str) -> str:
		url, text = await b.url(), await b.text()
		passage = slice_passage(text, starts_with, ends_with)
		if passage.startswith('!'):
			raise ToolError(passage[1:])
		mine = [m for m in kb.get('materials', []) if m['case'] == case]
		if mine and angle not in ('outcome', 'enrich') and case not in own:  # parallel missions pick keys independently
			raise ToolError(f'Case key {case!r} is already taken by another case. Use a distinct key that includes the record id (e.g. the registry or filing number).')
		own.add(case)
		if len(mine) >= MAX_MATERIALS and not any(f.get('case') == case for f in kb['facts']):  # complete cases may be enriched
			raise ToolError(
				f'Case {case} already has {len(mine)} materials and no outcome: record its outcome with record_fact or record_outcome (case="{case}") now (cut larger passages instead of many small ones).'
			)
		if any(passage in m['text'] for m in mine):
			return f'Already saved for case {case}: this passage is part of a saved material. Move on to its outcome.'
		kb.setdefault('materials', []).append({'id': f'm{len(kb["materials"])}', 'case': case, 'title': title, 'text': passage, 'url': url, 'at': now()})
		tally['materials'] = tally.get('materials', 0) + 1
		save()
		return f'Saved material for case {case}: {title} ({len(passage)} chars)'

	@tool('Record a realistic question a user might ask on this topic, why it is hard for a browser agent, and the page where its answer lives (if known).')
	async def record_task_idea(question: str, why_hard: str, answer_url: str = '') -> str:
		kb['ideas'].append({'question': question, 'why_hard': why_hard, 'answer_url': answer_url or await b.url(), 'angle': angle})
		save()
		return f'Recorded task idea: {question}'

	@tool('Record a useful source website for this topic (the CURRENT page) with a short note on what it contains and how to navigate it.')
	async def record_source(note: str) -> str:
		url, title = await b.url(), await b.title()
		kb['sources'].append({'url': url, 'title': title, 'note': note, 'angle': angle})
		save()
		return f'Recorded source {url}'

	return [record_fact, record_outcome, record_material, record_task_idea, record_source]


def source_health(kb: dict) -> str:
	"""Per start site: missions run, crashed, what they saved, and the miners' first note - so the next plan moves away
	from sources that yield nothing (an unstable search UI, a login wall) instead of repeating them."""
	by = {}
	for m in kb.get('missions', []):
		h = by.setdefault(urlparse(m['start_url']).netloc, {'n': 0, 'crashed': 0, 'materials': 0, 'outcomes': 0, 'note': ''})
		h['n'] += 1
		h['crashed'] += bool(m.get('error'))
		for k in ('materials', 'outcomes'):
			h[k] += m.get('recorded', {}).get(k, 0)
		h['note'] = h['note'] or next(iter(m.get('observations') or []), '')[:200]
	return '\n'.join(
		f'- {host}: {h["n"]} missions ({h["crashed"]} crashed) saved {h["materials"]} materials and {h["outcomes"]} outcomes' + (f' - miner note: {h["note"]}' if h['note'] else '')
		for host, h in by.items()
	)


async def plan(topic: str, n: int, model: str, angles: list[str] | None = None, sources: list[str] | None = None, health: str = '') -> list[Mission]:
	angles = angles or (['database', 'deep', 'interaction', 'longtail'] if sources else None)
	pool = [(k, ANGLES[k]) for k in angles] if angles else list(ANGLES.items())
	angles = [pool[i % len(pool)] for i in range(max(1, n))]
	sys = 'You plan web research missions for browser agents. Each mission gets a concrete, real https start URL (never a search engine) that is likely to work without login or captcha. Missions must cover DIFFERENT entities/sub-areas of the topic (name at least 3 distinct entities per mission, no entity in two missions) so the resulting benchmark is diverse.'
	if sources:  # the user knows where the data lives: every source gets missions, start URLs come from them
		sys += (
			' The user has named the sources where this data lives: start every mission on (a page of) one of these sources, spread missions evenly over them, and plan missions that go deep into each one.'
			' SOURCE PAGES shows what each source returns to a plain fetch: when it lists records (ids, names, links), give every mission its OWN records by id or name with their direct URLs - two missions must never get the same record.'
		)
		pages = await asyncio.gather(*(http_capture(u) for u in sources))  # free, and it lets the plan name concrete records instead of guessing filters
		topic += (
			'\nSOURCES:\n'
			+ '\n'.join(sources)
			+ '\n\nSOURCE PAGES (plain fetch, truncated):\n'
			+ '\n\n'.join(f'[{u}]\n{p["text"][:6000]}' for u, p in zip(sources, pages) if p.get('text'))
		)
	if health:
		sys += ' SOURCE HEALTH below reports what earlier missions saved per site: move missions away from sites that crash or save no complete cases, to other official pages with the same data (record pages, report or docket pages, bulk downloads, APIs).'
	user = (
		(f'SOURCE HEALTH from earlier missions:\n{health}\n\n' if health else '')
		+ f'Topic: {topic}\n\nWrite exactly one mission per line below (angles may repeat - then pick different entities). Make instructions specific to the topic.\n'
		+ '\n'.join(f'- angle `{k}`: {v}' for k, v in angles)
	)
	missions = []
	for _ in range(3):  # planners sometimes stop early: ask for the rest
		p, _, _ = await call(
			model,
			sys,
			user + (f'\n\nALREADY PLANNED (write only the {n - len(missions)} remaining, different ones):\n' + '\n'.join(m.instructions for m in missions) if missions else ''),
			Plan,
		)
		missions += p.missions
		if len(missions) >= n:
			break
	return missions[:n]


def outcome_missions(kb: dict, skill: str) -> list[Mission]:
	"""One mission per case that has materials but no outcome yet, starting where its material was cut."""
	done = {f.get('case') for f in kb['facts']} | {
		x.group(1) for m in kb.get('missions', []) if m.get('angle') == 'outcome' and not m.get('error') for x in [re.search(r'Case `([^`]+)`', m.get('instructions', ''))] if x
	}  # one try per case: an outcome not yet published must not starve new mining
	first = {}
	for m in kb.get('materials', []):
		if m['case'] not in done:
			first.setdefault(m['case'], m)
	return [
		Mission(
			angle='outcome',
			start_url=m['url'],
			instructions=f'Case `{k}` already has its input material ("{m["title"]}", on the start page). Do not collect new cases: find where the TRUE OUTCOME of this same case is published (for a forecast: the later document that reports it; otherwise the figures that settle the answer) and record it verbatim: record_fact(case="{k}") for one value, record_outcome(case="{k}") for a results section or table. If no outcome exists yet, say so and stop.',
		)
		for k, m in first.items()
	]


def enrich_missions(kb: dict) -> list[Mission]:
	"""One mission per complete case: save the data its computations need that the saved materials lack."""
	facts = {}
	for f in kb['facts']:
		facts.setdefault(f.get('case'), f)
	return [
		Mission(
			angle='enrich',
			start_url=f['url'],
			instructions=f'Case `{k}` is complete, but its saved materials may lack the data a solver needs to compute answers (results tables, financial statements, appendices). Save those sections from this page (or the case\'s other pages) with record_material(case="{k}"): whole tables, not single values. Do not collect new cases.',
		)
		for k, f in facts.items()
		if k and any(m['case'] == k for m in kb.get('materials', []))
	]


def case_task(topic: str, skill: str, m: Mission) -> str:
	if m.angle in ('outcome', 'enrich'):
		return (
			f'You are completing a case for an environment that trains a model in "{topic}". The model is given a case\'s input material and must: {skill}\n'
			f'{m.instructions}\nStart at {m.start_url}. Record the outcome verbatim under the case key as given: record_fact for one value, record_outcome for a results section or table. When done, report what made the outcome hard to find.'
		)
	return (
		f'You are collecting real cases for an environment that trains a model in "{topic}".\nThe model will be given the input materials of a case and must: {skill}\n'
		f'Mission: {m.instructions}\nStart at {m.start_url}. For each case pick a short key (e.g. the entity name and year), save its input with record_material '
		'(the passage a solver needs, cut from the page itself), then record its true outcome under the same key, verbatim from the page that states it: record_fact for one value, record_outcome for a results section or table (do not compute or summarize - save what the page shows). '
		'Finish one case completely (material AND outcome) before starting the next: 2 complete cases are worth more than 6 inputs without outcomes. '
		'Aim for 3-6 complete cases about DIFFERENT entities. Skip a case if its materials already state the outcome, or if no outcome is published. '
		'When done, report your observations about where inputs and outcomes live and what made them hard to reach.'
	)


async def run_mission(m: Mission, topic: str, kb: dict, model: str, max_steps: int, headless: bool = True, save=lambda: None, skill: str | None = None) -> dict:
	tally = {}
	task = (
		case_task(topic, skill, m)
		if skill
		else (
			f'You are scouting the topic "{topic}" to build a benchmark of realistic web tasks.\nMission ({m.angle}): {m.instructions}\n'
			f'Start at {m.start_url}. Use record_source for good sources, record_fact for verifiable facts (quote verbatim from the current page), '
			'and record_task_idea for realistic hard questions (at least 3). Aim for 8-12 facts about several DIFFERENT entities from DIFFERENT pages, including some reachable only after several clicks. '
			'Facts must be short atomic values that will not change soon (no view counts, stars, "N days ago"). '
			'When done, report your observations about what makes this domain hard for a browser agent.'
		)
	)
	b = browser()
	try:  # a cancelled or crashed mission still stops its browser (a Modal sandbox bills until it stops); what it saved stays saved
		await b.start()
		h = await run_agent(
			model,
			task,
			b,
			ScoutSummary,
			make_tools(kb, m.angle, b, save, tally),
			max_steps,
			done_description='Finish the mission: report your observations about where the data lives and what made it hard to reach. Call exactly once, as the last action.',
			timeout=MISSION_TIMEOUT - 60,
		)
	finally:
		await b.kill()
	return {
		'angle': m.angle,
		'start_url': m.start_url,
		'instructions': m.instructions,
		'steps': h.steps,
		'done': h.done,
		'observations': h.output.observations if h.output else [],
		'urls': h.urls,
		'recorded': tally,
	}


async def scout(
	topic: str,
	out_dir: Path,
	n_missions: int = 4,
	model: str = 'sonnet46',
	max_steps: int = 30,
	headless: bool = True,
	concurrency: int = 4,
	plan_model: str = 'sonnet46',
	angles: list[str] | None = None,
	sources: list[str] | None = None,
	skill: str | None = None,
	enrich: bool = False,
) -> dict:
	out_dir.mkdir(parents=True, exist_ok=True)
	kb_path = out_dir / 'kb.json'
	kb = json.loads(kb_path.read_text()) if kb_path.exists() else {'topic': topic, 'created': now(), 'facts': [], 'ideas': [], 'sources': [], 'missions': []}
	seen = '\n'.join(dict.fromkeys([m['case'] for m in kb.get('materials', [])] + [f['claim'] for f in kb['facts']]))[:3000]  # another round: go wide, not over the same records
	goal = topic + (f'\nSKILL the environment trains (input -> outcome): {skill}' if skill else '')
	todo = (enrich_missions(kb) if enrich else outcome_missions(kb, skill))[:n_missions] if skill else []  # finish the cases already started before opening new ones
	n_new = max(n_missions - len(todo), len(sources or ()) if not todo else 0)
	missions = todo + (
		await plan(
			goal + (f'\nALREADY MINED (choose DIFFERENT entities/records):\n{seen}' if seen else ''),
			n_new,
			plan_model,
			angles or (['cases'] if skill else None),
			sources,
			source_health(kb),
		)
		if n_new
		else []
	)
	sem = asyncio.Semaphore(concurrency)
	save = lambda: write_atomic(out_dir / 'kb.json', json.dumps(kb, indent=1))  # noqa: E731

	async def one(m):
		async with sem:
			for attempt in range(3):
				try:
					kb['missions'].append(await asyncio.wait_for(run_mission(m, topic, kb, model, max_steps, headless, save, skill), MISSION_TIMEOUT))
					break
				except (Exception, asyncio.CancelledError) as e:  # one mission must not sink the rest
					if isinstance(e, asyncio.CancelledError) and asyncio.current_task().cancelling():
						raise  # a real outer cancellation (Ctrl-C) must propagate
					if attempt < 2 and re.search(r'CloudError|sandbox|rate limit|usage limit|overloaded|HTTP 429|HTTP 503', f'{type(e).__name__}: {e}', re.I):
						await asyncio.sleep(60)  # the browser or the model never started: wait, then retry
						continue
					kb['missions'].append({'angle': m.angle, 'start_url': m.start_url, 'instructions': m.instructions, 'error': f'{type(e).__name__}: {e}'})
					break
			save()

	await ui.gather((one(m) for m in missions), 'mine')
	kb['spend_usd'] = round(LEDGER.usd, 4)
	save()
	return kb
