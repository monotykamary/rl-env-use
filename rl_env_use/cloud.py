"""Modal: every browser and every open-weight model of a command runs in a Modal Sandbox of one ephemeral app.

Containers stop as soon as they are done, in four layers, so a crash or a kill cannot leave one billing:
1. the code that starts a sandbox terminates it in a `finally` (a mission's browser, a check's browser);
2. `shutdown()` terminates every sandbox still alive when the command ends (model servers, anything missed);
3. Modal stops each sandbox on its own after `idle_timeout` without traffic, and always at its hard `timeout`;
4. the app is ephemeral: when this process exits or dies, Modal stops the app and every sandbox in it.
`rl-env-use stop` stops any app of this tool still running in the workspace, with its sandboxes (a belt-and-braces sweep).
"""

import asyncio
import atexit
import os
import secrets
import time

APP_NAME = 'rl-env-use'
TAGS = {'app': APP_NAME}  # every sandbox carries it: the Modal dashboard filters by it
BHJ_REPO = 'https://github.com/monotykamary/browser-harness-js'
BHJ_COMMIT = '0e1b2b6ebae16207446b741d18e65acc0ae8f227'  # browser-harness-js 0.17.0 (guarded page / locator API)
NODE = '24.11.1'
VLLM = '0.24.0'
REPL_PORT, PROXY_PORT, VLLM_PORT = 9876, 8080, 8000
# $/s at Modal's list prices: per physical CPU core, per GiB of memory, per GPU. They price the ledger, not the invoice.
CPU_S, MEM_S = 0.0000131, 0.00000222
GPU_S = {'T4': 0.000164, 'L4': 0.000222, 'A10G': 0.000306, 'L40S': 0.000542, 'A100-40GB': 0.000583, 'A100-80GB': 0.000694, 'H100': 0.001097, 'H200': 0.001261, 'B200': 0.001736}
BROWSER_CPU, BROWSER_MEM = 1.0, 2048
BROWSER_IDLE = int(os.getenv('RLENV_BROWSER_IDLE_S', '300'))  # a browser nobody has called for 5 minutes is stopped by Modal
BROWSER_TIMEOUT = int(os.getenv('RLENV_BROWSER_TIMEOUT_S', '3600'))  # hard ceiling of one browser's life
LLM_IDLE = int(os.getenv('RLENV_LLM_IDLE_S', '600'))  # a model server with no requests for 10 minutes is stopped by Modal
LLM_TIMEOUT = int(os.getenv('RLENV_LLM_TIMEOUT_S', str(6 * 3600)))

# A person's Chrome as far as sites can tell: the headless build's "HeadlessChrome" user agent and webdriver flag get blocked
START_BROWSER = f"""UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/$(chromium --version | grep -oE '[0-9]+' | head -1).0.0.0 Safari/537.36"
chromium --headless=new --no-sandbox --disable-gpu --disable-dev-shm-usage --no-first-run --no-default-browser-check \\
  --disable-blink-features=AutomationControlled --user-agent="$UA" --lang=en-US --window-size=1280,1600 \\
  --remote-debugging-port=9222 --user-data-dir=/tmp/chrome ${{PROXY:+--proxy-server=$PROXY}} about:blank >/tmp/chrome.log 2>&1 &
for i in $(seq 1 200); do curl -fs http://127.0.0.1:9222/json/version >/dev/null && break; sleep 0.1; done
CDP_REPL_PORT={REPL_PORT} browser-harness-js --start >/dev/null
nohup socat TCP-LISTEN:{PROXY_PORT},fork,reuseaddr TCP:127.0.0.1:{REPL_PORT} >/dev/null 2>&1 &
for i in $(seq 1 100); do curl -fs http://127.0.0.1:{PROXY_PORT}/health >/dev/null && break; sleep 0.1; done"""


class CloudError(RuntimeError):
	pass


def configured() -> bool:
	"""A Modal token is set up (`modal token new` writes ~/.modal.toml; CI exports MODAL_TOKEN_ID/SECRET)."""
	return bool(os.getenv('MODAL_TOKEN_ID')) or os.path.isfile(os.path.expanduser('~/.modal.toml'))


def browser_image():
	import modal

	return (
		modal.Image.debian_slim(python_version='3.12')
		.apt_install('chromium', 'fonts-liberation', 'fonts-noto-cjk', 'fonts-noto-color-emoji', 'curl', 'git', 'ca-certificates', 'xz-utils', 'socat')
		.run_commands(
			f'curl -fsSL https://nodejs.org/dist/v{NODE}/node-v{NODE}-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1',
			f'git clone {BHJ_REPO} /opt/bhj && git -C /opt/bhj checkout {BHJ_COMMIT}',
			'ln -s /opt/bhj/skills/cdp/sdk/browser-harness-js /usr/local/bin/browser-harness-js',
		)
	)


# weights, torch.compile artifacts and FlashInfer's JIT kernels: kept across starts, so only the first one downloads and compiles
LLM_CACHES = {'/root/.cache/huggingface': 'huggingface-cache', '/root/.cache/vllm': 'vllm-cache', '/root/.cache/flashinfer': 'flashinfer-cache'}


def vllm_image():
	import modal

	return (
		modal.Image.from_registry('nvidia/cuda:12.9.0-devel-ubuntu22.04', add_python='3.12')
		.entrypoint([])
		.uv_pip_install(f'vllm=={VLLM}')
		.env({'HF_XET_HIGH_PERFORMANCE': '1', 'HF_HUB_CACHE': '/root/.cache/huggingface', 'VLLM_CACHE_ROOT': '/root/.cache/vllm'})
	)


class Live:
	"""One running sandbox and what it costs per second; `allocated` is the share already booked per request."""

	def __init__(self, sb, label: str, rate: float):
		self.sb, self.label, self.rate, self.t0, self.allocated = sb, label, rate, time.monotonic(), 0.0

	def spent(self) -> float:
		return (time.monotonic() - self.t0) * self.rate


_app = _ctx = _lock = None
LIVE: dict[str, Live] = {}


async def app():
	"""The command's ephemeral app, started on first use (no Modal round trip for commands that never need one)."""
	global _app, _ctx, _lock
	if _app is None:
		_lock = _lock or asyncio.Lock()
		async with _lock:
			if _app is None:
				import modal

				if not configured():
					raise CloudError('Modal is not set up: run `modal token new` (browsers and open-weight models run on Modal)')
				a = modal.App(APP_NAME)
				ctx = a.run()
				await ctx.__aenter__()
				_app, _ctx = a, ctx
	return _app


async def sandbox(
	*cmd: str, label: str, image, cpu: float, memory: int, gpu: str | None = None, timeout: int, idle_timeout: int, env: dict | None = None, volumes=None, **kw
) -> Live:
	import modal

	a = await app()
	sb = await modal.Sandbox.create.aio(
		*cmd,
		app=a,
		image=image,
		cpu=cpu,
		memory=memory,
		gpu=gpu,
		timeout=timeout,
		idle_timeout=idle_timeout,
		env=env or {},
		volumes=volumes or {},
		tags=TAGS | {'kind': label.split()[0]},
		**kw,
	)
	kind, _, n = (gpu or '').partition(':')
	rate = cpu * CPU_S + memory / 1024 * MEM_S + GPU_S.get(kind, GPU_S['H100'] if gpu else 0.0) * int(n or 1)
	live = LIVE[sb.object_id] = Live(sb, label, rate)
	return live


async def terminate(live: Live | None) -> None:
	"""Stop a sandbox now and book what it cost beyond the requests that already carried their share."""
	if live is None or LIVE.pop(live.sb.object_id, None) is None:
		return
	from .llm import LEDGER

	if live.label.startswith('llm'):  # commit what this start downloaded and compiled before the volumes detach
		try:
			p = await asyncio.wait_for(live.sb.exec.aio('sync', *LLM_CACHES, timeout=60), 10)
			await asyncio.wait_for(p.wait.aio(), 70)
		except (Exception, asyncio.CancelledError):
			pass
	try:
		await asyncio.wait_for(live.sb.terminate.aio(), 30)
	except (Exception, asyncio.CancelledError):
		try:  # the async path died with the loop: the sync client runs on its own thread
			live.sb.terminate()
		except Exception:
			pass
	LEDGER.add(f'modal {live.label}', {'reported_cost': max(0.0, live.spent() - live.allocated)})


async def exec_ok(live: Live, script: str, timeout: int = 120) -> str:
	p = await live.sb.exec.aio('bash', '-c', script, timeout=timeout)
	out, err = await p.stdout.read.aio(), await p.stderr.read.aio()
	if await p.wait.aio():
		raise CloudError(f'sandbox command failed: {(err or out)[-400:]}')
	return out


async def browser_sandbox(proxy: str | None = None) -> tuple[Live, str, str]:
	"""A headless Chromium driven by browser-harness-js. Returns (sandbox, REPL url, bearer token): the REPL is reached
	through Modal's authenticated proxy, never a public port."""
	live = await sandbox('sleep', 'infinity', label='browser', image=browser_image(), cpu=BROWSER_CPU, memory=BROWSER_MEM, timeout=BROWSER_TIMEOUT, idle_timeout=BROWSER_IDLE)
	try:
		await exec_ok(live, (f'export PROXY={proxy!r}\n' if proxy else '') + START_BROWSER)
		cred = await live.sb.create_connect_token.aio(port=PROXY_PORT)
		return live, cred.url.rstrip('/'), cred.token
	except BaseException:
		await terminate(live)
		raise


async def vllm_sandbox(model: str, gpu: str, args: list[str]) -> tuple[Live, str, str]:
	"""An OpenAI-compatible vLLM server for one Hugging Face model. Returns (sandbox, base url, API key): an encrypted
	tunnel (long generations outlast an HTTP proxy's limits) guarded by a random vLLM API key. Weights and compile
	caches live in Modal Volumes, so only the first start downloads."""
	import modal

	key = secrets.token_urlsafe(24)
	vols = {path: modal.Volume.from_name(name, create_if_missing=True) for path, name in LLM_CACHES.items()}
	cmd = ['vllm', 'serve', model, '--served-model-name', model, '--host', '0.0.0.0', '--port', str(VLLM_PORT), '--api-key', key, *args]
	live = await sandbox(
		*cmd, label=f'llm {model}', image=vllm_image(), cpu=4.0, memory=32768, gpu=gpu, timeout=LLM_TIMEOUT, idle_timeout=LLM_IDLE, volumes=vols, encrypted_ports=[VLLM_PORT]
	)
	try:
		tunnel = (await live.sb.tunnels.aio(timeout=120))[VLLM_PORT]
		return live, tunnel.url.rstrip('/'), key
	except BaseException:
		await terminate(live)
		raise


async def shutdown() -> None:
	"""End of a command: stop every sandbox still alive, then the ephemeral app."""
	global _app, _ctx, _lock
	await asyncio.gather(*(terminate(x) for x in list(LIVE.values())), return_exceptions=True)
	_lock = None  # bound to this event loop: the next command's loop makes its own
	if _ctx is not None:
		ctx, _app, _ctx = _ctx, None, None
		try:
			await asyncio.wait_for(ctx.__aexit__(None, None, None), 60)
		except (Exception, asyncio.CancelledError):
			pass


@atexit.register
def _sweep() -> None:
	"""Library use without `shutdown()`: the sync client still reaches Modal after the event loop is gone."""
	for live in list(LIVE.values()):
		LIVE.pop(live.sb.object_id, None)
		try:
			live.sb.terminate()
		except Exception:
			pass


def stop_all() -> list[str]:
	"""Stop every running app of this tool in the Modal workspace, and with it every sandbox in it, from any process
	(`rl-env-use stop`). This includes a command still running in another terminal; other apps are never touched."""
	import json
	import shutil
	import subprocess
	import sys
	from pathlib import Path

	cli = str(Path(sys.executable).with_name('modal')) if Path(sys.executable).with_name('modal').exists() else shutil.which('modal') or 'modal'
	out = subprocess.run([cli, 'app', 'list', '--json'], capture_output=True, text=True, check=True).stdout
	running = [a['app_id'] for a in json.loads(out) if a.get('description') == APP_NAME and a.get('state') != 'stopped']
	for app_id in running:
		subprocess.run([cli, 'app', 'stop', app_id], capture_output=True, check=True)
	return running
