#!/usr/bin/env python3
"""
mcpkit.client: Minimal client for the MCP-like stdio server.

CLI:
  python -m mcpkit.client list
  python -m mcpkit.client call --tool add --args '{"a":1,"b":2}'
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import typing as t
from dataclasses import dataclass

JSON = t.Dict[str, t.Any]


@dataclass
class MCPClient:
    server_cmd: list[str] | None = None
    process: asyncio.subprocess.Process | None = None
    _id: int = 0

    async def __aenter__(self):
        if self.server_cmd is None:
            self.server_cmd = [sys.executable, "-m", "mcpkit.server", "--name", "mcpkit_demo"]
        self.process = await asyncio.create_subprocess_exec(
            *self.server_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "shutdown"})
            await self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "exit"})
        except Exception:
            pass
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except Exception:
                pass

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _send(self, msg: JSON) -> JSON:
        assert self.process and self.process.stdin and self.process.stdout
        data = (json.dumps(msg) + "\n").encode("utf-8")
        self.process.stdin.write(data)
        await self.process.stdin.drain()
        line = await self.process.stdout.readline()
        if not line:
            raise RuntimeError("Server closed pipe")
        resp = json.loads(line.decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(f"RPC error: {resp['error']}")
        return resp

    async def initialize(self) -> JSON:
        return await self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "initialize", "params": {}})

    async def list_tools(self) -> list[JSON]:
        resp = await self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list"})
        return resp["result"]["tools"]

    async def call(self, name: str, **kwargs) -> list[JSON]:
        resp = await self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": kwargs},
        })
        return resp["result"]["content"]


def call(tool: str, args: dict | None = None, server_cmd: list[str] | None = None) -> str:
    async def _run():
        async with MCPClient(server_cmd=server_cmd) as cli:
            await cli.initialize()
            content = await cli.call(tool, **(args or {}))
            for part in content:
                if part.get("type") == "text":
                    return part.get("text", "")
            return ""
    return asyncio.run(_run())


def _cli():
    ap = argparse.ArgumentParser(description="Client for mcpkit server")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List available tools")

    ccall = sub.add_parser("call", help="Call a tool by name")
    ccall.add_argument("--tool", required=True, help="Tool name")
    ccall.add_argument("--args", default="{}", help='JSON dict of arguments, e.g. \'{"a":1,"b":2}\'')
    ccall.add_argument("--server-cmd", default=None, help="Override server command (JSON list)")

    args = ap.parse_args()

    async def _amain():
        server_cmd = None
        if getattr(args, "server_cmd", None):
            server_cmd = json.loads(args.server_cmd)
        async with MCPClient(server_cmd=server_cmd) as cli:
            await cli.initialize()
            if args.cmd == "list":
                tools = await cli.list_tools()
                print(json.dumps(tools, indent=2, ensure_ascii=False))
            elif args.cmd == "call":
                kwargs = json.loads(args.args)
                content = await cli.call(args.tool, **kwargs)
                print(json.dumps(content, indent=2, ensure_ascii=False))
