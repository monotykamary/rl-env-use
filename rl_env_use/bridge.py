"""MCP stdio server for the CLI agents (Claude Code, Codex): it lists one agent run's tools and relays every call to the
rl-env-use process that owns the browser and the kb, over a loopback socket with a one-time token. Standard library
only and run as a script (no package import), so it starts in milliseconds.

usage: python bridge.py <port> <token>
"""

import json
import socket
import sys


def ask(port: int, token: str, req: dict) -> dict:
	with socket.create_connection(('127.0.0.1', port), timeout=900) as s:
		s.sendall((json.dumps(req | {'token': token}) + '\n').encode())
		with s.makefile('r', encoding='utf-8') as f:
			line = f.readline()
	if not line:  # the run is over, or the token is not this run's
		raise ConnectionError('the rl-env-use run refused or ended this call')
	return json.loads(line)


def main() -> None:
	port, token = int(sys.argv[1]), sys.argv[2]
	for line in sys.stdin:
		try:
			msg = json.loads(line)
		except ValueError:
			continue
		if msg.get('id') is None:  # a notification (initialized, cancelled): nothing to answer
			continue
		method, params, out = msg.get('method'), msg.get('params') or {}, {'jsonrpc': '2.0', 'id': msg['id']}
		try:
			if method == 'initialize':
				out['result'] = {'protocolVersion': params.get('protocolVersion', '2025-06-18'), 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'rlenv', 'version': '1'}}
			elif method == 'tools/list':
				out['result'] = {'tools': ask(port, token, {'op': 'list'}).get('tools', [])}
			elif method == 'tools/call':
				r = ask(port, token, {'op': 'call', 'name': params.get('name'), 'args': params.get('arguments') or {}})
				out['result'] = {'content': [{'type': 'text', 'text': r.get('text', '')}], 'isError': bool(r.get('is_error'))}
			elif method == 'ping':
				out['result'] = {}
			else:
				out['error'] = {'code': -32601, 'message': f'unknown method {method}'}
		except Exception as e:  # the run ended or the socket broke: the agent sees an error, not a hang
			out['error'] = {'code': -32000, 'message': f'{type(e).__name__}: {e}'}
		sys.stdout.write(json.dumps(out) + '\n')
		sys.stdout.flush()


if __name__ == '__main__':
	main()
