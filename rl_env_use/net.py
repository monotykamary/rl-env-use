"""HTTP calls that a cancellation cannot freeze: requests run on worker threads (cancelling an awaited httpx request closes
its TLS stream in a shielded section, and a peer that never finishes the close froze whole climbs - every slot, every
timeout). A timeout or cancellation returns at once; the thread finishes or times out on its own."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

_pool: ThreadPoolExecutor | None = None
_local = threading.local()


def _client(timeout: float):
	import httpx

	c = getattr(_local, 'client', None)
	if c is None or c.is_closed:
		c = _local.client = httpx.Client(timeout=httpx.Timeout(timeout, connect=30), follow_redirects=False)
	return c


def _post(url: str, headers: dict, body, timeout: float):
	kw = {'json': body} if isinstance(body, (dict, list)) else {'content': body}
	return _client(timeout).post(url, headers=headers, timeout=timeout, **kw)


async def post(url: str, body, headers: dict | None = None, timeout: float = 600):
	"""POST `body` (a dict/list goes as JSON, a str as-is); returns the httpx.Response."""
	global _pool
	if _pool is None:
		_pool = ThreadPoolExecutor(64, thread_name_prefix='rlenv-http')
	return await asyncio.get_running_loop().run_in_executor(_pool, _post, url, headers or {}, body, timeout)
