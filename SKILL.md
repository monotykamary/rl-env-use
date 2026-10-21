---
name: rl-env-use
description: Build an RL environment / eval that trains or measures a skill in any domain (finance, medicine, law, math, science, a niche industry) from real cases that browser agents (browser-harness-js on Modal) mine from the web, then baseline and hillclimb a model on it. Use when the user knows a domain and what a model should be able to do in it, but not how to get the data or build the environment.
---

# rl-env-use

Anthropic's [build-eval -> hillclimb](https://claude.dev/blog/automating-eval-design-and-hillclimbing/) workflow with
real inputs. If the claude-api skill is available, its `shared/evals/build-eval.md` and `shared/evals/eval-hillclimb.md`
are the method. A case = the materials the model gets + its true outcome, published elsewhere or later. Browser agents
mine both. The model under test reads the materials and answers without a browser.

No API keys: Claude models run through `claude -p`, GPT models through `codex exec`, browsers and open-weight models
(qwen-35b, qwen-27b, `modal:<hf-org>/<model>`) on Modal. Every Modal container stops when its job ends, when the
command ends, or when Modal sees it idle; `rl-env-use stop` sweeps any left over. Never leave a command of this tool
running in the background without telling the user: it holds Modal containers while it runs.

Commands: `rl-env-use ...` (or `python -m rl_env_use ...`). Every env is one folder, `./envs/<name>/`.

## Where things are saved

```
envs/<name>/
  spec.json        topic, skill, sources, conventions, budget; `create` adds spend and result
  kb.json          what the agents mined: materials, quote-verified outcome facts, missions
  tasks.jsonl      every drafted task with gold and checks (the env serves the ones that pass)
  audit.json       latest audit (also printed by `report`)
  reviews.jsonl    your verdicts and gold fixes from the review app
  models/          model comparisons, one slot per model (named in <slot>/change.md)
  climb-<model>/   a hillclimb: baseline, v1..vN (change.md = the edit, prompt.txt = full prompt), _state.json, report.html
  hillclimb/       runs by hand; ablation/: runs with materials withheld (--blind)
  <flow>/<slot>/   results.jsonl (graded rows), errors.jsonl (infra failures), traces/ (transcripts)
```
Slots are `baseline`, `v1`, `v2`, ...; `change.md`'s first line is the report's column title. Traces, trajectory/ and
logs are git-ignored.

## Autonomous (default when the user wants it hands-off)

1. `create "<domain, skill, sources, constraints in one line>" [--tasks 150] [--budget 40]`. A planner writes the spec
   (skill, sources, conventions, an outcome-blind sampling rule). Mining rounds then run until the task target, the
   budget, or two rounds with no new tasks. Each round: mine -> write -> verify -> panel -> harden -> verify -> panel.
   - Derived numbers are computed by code from an expression over numbers in the materials.
   - The frontier panel (opus55, gpt-astra) measures difficulty. All-pass tasks go back to the drafter and
     come back harder on the same case, up to two levels. Forecasts must move with the materials and must not be
     answerable without them. At most 60% of served forecasts resolve one way.
   - When the panel saturates derive tasks (on SEC filings it solves nearly every computation), later rounds write
     forecasts. Derive tasks stay useful for smaller policies: pass `--panel` with the policy family.
   - Mining is fastest when the line names a URL that lists records (a registry API query, a search page).
   - Ctrl-C stops cleanly (every Modal container is stopped); `create --name <env>` resumes with its spend so far.
2. `climb envs/<name> --model M`: the hillclimb loop in `climb-<model>/`. Split by case, A/A noise from rep halves,
   refuses saturated or underpowered setups, K edits per round screened on train, kept only on a paired case-bootstrap
   test gain, failure audit on stalls, final baseline-vs-best on fresh reps. Report the final line and
   `climb-<model>/report.html` as they are: "within noise, do not merge" is a result.

On a terminal both show a live board (stage bar with ETA, spend vs budget, kept tasks; for climb: baseline, best,
rounds). In a log they print status lines.

## Step by step (when the user wants to review)

Ask every decision with AskUserQuestion, your pick first as "(Recommended)". Wait for a clear yes at each sign-off.

**Step 0: what is evaluated.** Ask: (1) the skill as input -> outcome ("given a trial's protocol, predict whether it met
its primary endpoint"), one per env; (2) where inputs and outcomes live (2-6 URLs), ideally separated in place or time;
(3) a proxy for geo-restricted sources, if any. Then
`new <name> --topic "<domain>" --skill "<input -> outcome>" --sources URL.. [--proxy http://host:port]`.
Without `--skill` it is a browse-task env (the model must find answers on the live web).

**Step 1: mine the cases (sign-off 1).** `build envs/<name>` mines (~$0.15-0.40 per mission) and stops before any task
exists. Show the look output (`build envs/<name> --only look` reprints it). The user checks `review envs/<name>` -> Data
tab. Too few cases: `build --only scout`. Tasks that compute from tables need them in the materials:
`build --only scout --enrich`. SIGN-OFF: "Are these the cases the environment should be built from?"

**Step 2: tasks and grading (sign-off 2).** `build envs/<name>` again writes and checks tasks: gold from outcome quotes,
no leak into prompt or materials, closed-book memory check, live evidence, a critic. Grading is programmatic (number /
date / set / exact / contains / prob), self-tested against null, foreign and perturbed golds. Show 5 tasks (Tasks tab).
SIGN-OFF: "Would you grade any of these differently?"

**Step 3: pilot and cost (sign-off 3).** `run envs/<name> --model M --limit 5`, then `run ... --dry-run` prices the full
run from measured cost per episode. SIGN-OFF on that number.

**Step 4: baseline.** `run envs/<name> --model M --flow models` per model (each gets its own slot). `report envs/<name>`
regrades, audits and writes each `report.html`. `inverted` cases usually mean a task or grader bug. Headroom is relative
to the policy: frontier-saturated derive tasks can still train a smaller model; for frontier headroom use forecasts on
lesser-known records. Ask whether to commit the env.

**Step 5: hillclimb.** `climb envs/<name> --model M`, or by hand: `split envs/<name>`, then per round
`run envs/<name> --model M --variant vN --system prompt.txt`. Label failures in the Traces tab, `codify`, then `judges`
(trusted once they agree with the user's labels). `export envs/<name>` writes the dataset for an RL trainer;
`from rl_env_use import Env` serves it (`reset`, `reward`).

## Optional: measure the edge

`compare envs/<name> --missions 8` reruns the mining with vanilla Claude Code (WebSearch + WebFetch, same model and
instructions) and pushes both arms through the same synth + verify.
