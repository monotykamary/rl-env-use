---
name: rl-env-use-install
description: Install rl-env-use into the current agent (Claude Code, Codex, Hermes, Openclaw, etc.) and check that Modal and the claude / codex CLIs are signed in, so the user can build an environment right away. No API keys are needed.
---

# rl-env-use install

Use this file only for first-time install or reconnect. For daily use, read `SKILL.md`.

## What you're doing

You're setting up a tool that turns one line ("predict whether a clinical trial met its primary endpoint from its
registered protocol") into an RL environment built from real cases that browser agents mine from the web, and then
hillclimbs a model on it. After install the user says what skill they want an environment for; you run it by reading
`SKILL.md`.

Five things must exist on this machine:

1. The `rl-env-use` repo cloned somewhere stable, installed into its own virtual environment (Python 3.11+).
2. A Modal token. Every browser (headless Chromium driven through browser-harness-js) and every open-weight model
   (vLLM on a GPU) runs in a Modal sandbox. Each sandbox stops when its job ends.
3. At least one signed-in model CLI: Claude Code (`claude`) for the Claude models, which are the default for the
   planner, task writer, critic, miners and policies. Codex (`codex`) for the GPT models, which sit on the default
   difficulty panel next to Opus.
4. Optionally Node.js (>= 18) on `$PATH`, which writes the HTML reports. Without it everything works and the reports
   are skipped.
5. The current agent can discover `SKILL.md` (a skills directory, or an import in its `CLAUDE.md`-equivalent).

## Install prompt contract

- Do everything yourself. Ask the user only to sign in where something is not signed in (Modal, claude, codex).
- Prefer a stable clone path like `~/Developer/rl-env-use` (not `/tmp`, not `~/Downloads`).
- Never spend credits during install. Every check below is free, and none starts a Modal container.

## Steps

### 1. Clone and install

```bash
test -d ~/Developer/rl-env-use || git clone https://github.com/monotykamary/rl-env-use ~/Developer/rl-env-use
cd ~/Developer/rl-env-use
command -v uv >/dev/null && uv venv && uv pip install -e '.[test]' || (python3 -m venv .venv && .venv/bin/pip install -e '.[test]')
.venv/bin/python -m pytest -q   # offline, no accounts needed, ~10 s
```

If the repo is already there, `git pull --ff-only` and re-run the install line.

### 2. Register the skill

- **Claude Code**: `mkdir -p ~/.claude/skills && ln -sfn ~/Developer/rl-env-use ~/.claude/skills/rl-env-use`
- **Codex**: `mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills" && ln -sfn ~/Developer/rl-env-use "${CODEX_HOME:-$HOME/.codex}/skills/rl-env-use"`
- **Another agent**: symlink the repo into its skills directory as `rl-env-use`, or point its system prompt at
  `~/Developer/rl-env-use/SKILL.md`.

Put the CLI on the user's path so it runs from any folder: `ln -sfn ~/Developer/rl-env-use/.venv/bin/rl-env-use ~/.local/bin/rl-env-use`
(any directory on `$PATH` works).

### 3. Accounts

Check each one and stop at the first hit:

```bash
.venv/bin/modal profile current                                   # a profile name = Modal is set up
command -v claude && echo 'Reply with: ok' | claude -p --model haiku --tools '' --setting-sources ''   # ok = Claude is signed in
command -v codex && codex login status                            # "Logged in" = Codex is signed in
```

- **No Modal token**: ask the user to run `! ~/Developer/rl-env-use/.venv/bin/modal token new` (it opens a browser to sign in).
- **No `claude`**: Claude Code is required for the default models. Ask the user to install and sign in
  (https://claude.com/claude-code), or, if they prefer the API, to put `ANTHROPIC_API_KEY=...` in
  `~/Developer/rl-env-use/.env` (`cp -n .env.example .env && chmod 600 .env`; never echo the key back, never commit `.env`).
- **No `codex`**: optional. Without it, the GPT models are unavailable; pass `--panel opus55 sonnet55` to `create`.

Nothing else is needed: there is no OpenRouter or Browser Use key.

### 4. Verify end-to-end (free)

```bash
.venv/bin/rl-env-use run envs/trial-forecast --model opus55 --flow models --dry-run   # prices a run from the shipped env's measurements; writes nothing
```

### 5. Hand off

Tell the user, in one short message:

- where it is installed;
- that a good first message is *"build me an RL environment for <skill> from <where the data lives>"*;
- that a build costs real usage: Claude and Codex calls count against their accounts, and browsers and open-weight
  models run on Modal. The budget counts the API-equivalent price plus Modal time; the shipped envs cost $20-75 each,
  and `create` stops at its `--budget`, $40 by default;
- that every Modal container stops by itself when its job ends, and `rl-env-use stop` sweeps any left over;
- that everything lands in `./envs/<name>/`.

## Keeping it current

`cd ~/Developer/rl-env-use && git pull --ff-only && uv pip install -e .` (or `.venv/bin/pip install -e .`).
