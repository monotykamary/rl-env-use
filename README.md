<p align="center">
  <img src="static/rl-env-use.gif" alt="rl-env-use: one line in, a verified RL environment out" width="100%">
</p>

# rl-env-use

Introducing **rl-env-use**: RL environments for anyone and any domain.

Describe a skill in one line. Get a verified environment that frontier models do not saturate, and a hillclimbed prompt
for the model you train. Built on Anthropic's [Automating eval design and hillclimbing](https://claude.dev/blog/automating-eval-design-and-hillclimbing/)
workflow, with one change: the tasks are real cases that browser agents mine from the web. Each agent drives a
headless Chromium in a [Modal](https://modal.com) sandbox through [browser-harness-js](https://github.com/monotykamary/browser-harness-js).

No API keys are needed. Claude models run through your Claude Code login (`claude -p`), GPT models through your Codex
login (`codex exec`), and open-weight policies (Qwen) on a Modal GPU with vLLM. Every Modal container stops when its
job is done (see [Costs and cleanup](#costs-and-cleanup)).

The measurements below were made with this project's earlier Browser Use harness. They show what a real browser
finds that WebSearch + WebFetch does not:

| Missions | Claude Code | Browser agents |
|----------|-------------|-------------|
| 8 x clinical trial results | 0 of 86 facts verified | **17 verified**, 4 complete cases |
| 6 x clinical trial results, same model both sides | 0 of 43 | **3 verified**, 2 complete cases |
| 8 x human genetic variants (UniProt, gnomAD, GTEx) | 6 of 82 | **40 verified** |

## What it does

- **Mines real cases**: browser agents find the materials (a trial protocol, a 10-Q) and the outcome published later
- **Writes hard tasks**: gold answers computed by code, anything a frontier panel solves gets rewritten harder
- **Rejects bad tasks**: leaked answers, memorized outcomes, ambiguous prompts
- **Hillclimbs** a model's system prompt with a train/test split, a noise floor and paired confidence intervals
- **Exports** the dataset for an RL trainer, or serves it in code with `Env`

## Setup prompt

For Claude Code / Codex / OpenCode:

```text
Set up https://github.com/monotykamary/rl-env-use for me.

Read install.md first to install this repo, register the skill, and check that Modal and the claude / codex CLIs are signed in - ask me to sign in where one is not. Then read SKILL.md for daily usage. After install, don't build anything on your own - just tell me it's ready and wait for me to name a skill.
```

Then ask:

> build me an RL environment for predicting whether a clinical trial met its primary endpoint, from its ClinicalTrials.gov registration

For humans:

```bash
rl-env-use create "predict whether a completed trial met its primary endpoint from its registered protocol; ClinicalTrials.gov" --budget 40
rl-env-use climb envs/<name> --model qwen-35b
```

## Results

The shipped example, [`envs/trial-forecast`](envs/trial-forecast): 35 forecasts from 29 ClinicalTrials.gov trials whose
results were posted since July 2026, after the models' training data. Built for $72 (on the earlier Browser Use + OpenRouter stack).

<p align="center"><img src="static/chart-scores.png" alt="held-out test scores: GPT-6 Astra 0.81, climbed Qwen 0.78, Opus 5.5 0.77" width="100%"></p>

A climbed Qwen 3.6 35B-A3B goes from 0.53 to 0.78 on tasks it never trained on (+0.25, 95% CI +0.07 to +0.45), level
with Opus 5.5. Always answering 50% scores 0.75, so most of the gain is calibration: the model stopped answering 0 or 1.
The climb kept 1 of 12 proposed edits (base-rate priors per question type); every round is in
[`climb-qwen-35b/`](envs/trial-forecast/climb-qwen-35b).

## Manual install

Python 3.11+, a [Modal](https://modal.com) account, and at least one signed-in model CLI: [Claude Code](https://claude.com/claude-code)
(`claude`) and/or [Codex](https://github.com/openai/codex) (`codex`). Node.js 18+ is optional; it writes the HTML reports.

```bash
git clone https://github.com/monotykamary/rl-env-use ~/Developer/rl-env-use
cd ~/Developer/rl-env-use
uv venv && uv pip install -e '.[test]'
.venv/bin/modal token new                # once: browsers and open-weight models run on Modal
.venv/bin/python -m pytest -q           # offline, ~10 s
```

## Where things run

| What | Where | Billed to |
|------|-------|-----------|
| Browser agents (miners, browse-task policies), live quote checks | headless Chromium in a Modal sandbox, driven through browser-harness-js | Modal (about $0.06 per browser-hour) |
| Claude models (planner, task writer, critic, panel, policies) | `claude -p`, none of your settings, hooks or MCP servers | your Claude account |
| GPT models (panel, adjudication, policies) | `codex exec`, read-only, none of your config | your ChatGPT account |
| Open-weight policies (`qwen-35b`, `qwen-27b`, `modal:<hf-org>/<model>`) | vLLM on a Modal GPU (H100 by default) | Modal |
| Anthropic API (optional) | `ANTHROPIC_API_KEY` in `.env`, used only as a fallback route for Claude models | Anthropic |

A browser agent's model can be any of these. Claude and Codex run their own agent loop and get the browser tools
through a small MCP bridge back to rl-env-use. Modal-served models run in rl-env-use's own tool-calling loop.

## Costs and cleanup

Every Modal container has four layers of shutdown, so a crash or a kill cannot leave one billing:

1. Each mission, check or episode stops its browser in a `finally` as soon as it ends.
2. At the end of every command (including Ctrl-C and `kill`), every container still alive is stopped, model servers included.
3. Modal stops a container on its own when it is idle (browsers after 5 minutes, model servers after 10) and always at
   its hard timeout (1 h per browser, 6 h per model server). Set `RLENV_BROWSER_IDLE_S`, `RLENV_LLM_IDLE_S`,
   `RLENV_BROWSER_TIMEOUT_S` or `RLENV_LLM_TIMEOUT_S` to change these.
4. All containers belong to an ephemeral Modal app, which Modal stops when the command's process exits or dies.

`rl-env-use stop` stops every rl-env-use app still running in your Modal workspace, with all its containers (a command
still running in another terminal included; other Modal apps are never touched). The budget (`--budget`)
counts the API-equivalent price of every model call (`claude -p` reports it; codex tokens are priced at list rates)
plus each container's Modal time.

See [`SKILL.md`](./SKILL.md) for the full workflow and where every file is saved.

## License

MIT, see [LICENSE](LICENSE).
