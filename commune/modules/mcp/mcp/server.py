#!/usr/bin/env python3
"""
mcpkit.server: Turn any Python class into a minimal MCP-like server over stdio.

Usage:
  python -m mcpkit.server --name demo_server
  python -m mcpkit.server --name myserver --target "package.module:MyClass"
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import io
import json
import sys
import typing as t
from dataclasses import dataclass, field

JSON = t.Dict[str, t.Any]


def _resolve_target(target: str | None):
    if target is None:
        from .demo import Demo
        return Demo()
    if ":" not in target:
        raise ValueError("--target must be 'module:AttrPath'")
    mod_name, attr_path = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if inspect.isclass(obj):
        return obj()
    return obj


def _is_public_method(name: str, member: t.Any) -> bool:
    if name.startswith("_"):
        return False
    if not callable(member):
        return False
    if isinstance(member, (staticmethod, classmethod)):
        return False
    return True


def _type_to_schema(tp: t.Any) -> JSON:
    origin = t.get_origin(tp)
    args = t.get_args(tp)

    def base_schema(py):
        if py is str:
            return {"type": "string"}
        if py is int:
            return {"type": "integer"}
        if py is float:
            return {"type": "number"}
        if py is bool:
            return {"type": "boolean"}
        if py in (dict,):
            return {"type": "object"}
        if py in (list,):
            return {"type": "array", "items": {}}
        return {}

    if origin is t.Union and len(args) == 2 and type(None) in args:
        other = args[0] if args[1] is type(None) else args[1]
        sub = _type_to_schema(other)
        return {"anyOf": [sub or {}, {"type": "null"}]}

    if origin is t.Union:
        return {"anyOf": [(_type_to_schema(a) or {}) for a in args]}

    if origin in (list, t.List):
        item = _type_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin in (dict, t.Dict):
        val = _type_to_schema(args[1]) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": val}

    if origin is t.Literal:
        return {"enum": list(args)}

    if origin is t.Annotated and args:
        return _type_to_schema(args[0])

    return base_schema(tp)


def _build_params_schema(func: t.Callable) -> JSON:
    sig = inspect.signature(func)
    props: JSON = {}
    required: t.List[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ann = param.annotation if param.annotation is not inspect._empty else t.Any
        schema = _type_to_schema(ann) or {}
        props[name] = schema
        if param.default is inspect._empty:
            required.append(name)
    schema: JSON = {
        "type": "object",
        "properties": props,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: JSON
    func: t.Callable
    is_async: bool


@dataclass
class MCPClassServer:
    name: str = "mcpkit_server"
    target: t.Any = field(default_factory=lambda: None)
    stdin: io.TextIOBase = field(default_factory=lambda: sys.stdin)
    stdout: io.TextIOBase = field(default_factory=lambda: sys.stdout)

    def __post_init__(self):
        self._tools = self._collect_tools()
        self.loop: asyncio.AbstractEventLoop | None = None

    def _collect_tools(self) -> t.Dict[str, ToolSpec]:
        tools: dict[str, ToolSpec] = {}
        for name, member in inspect.getmembers(self.target):
            if not _is_public_method(name, member):
                continue
            try:
                desc = inspect.getdoc(member) or f"{self.target.__class__.__name__}.{name}"
                schema = _build_params_schema(member)
                is_async = inspect.iscoroutinefunction(member)
                tools[name] = ToolSpec(
                    name=name,
                    description=desc,
                    params_schema=schema,
                    func=member,
                    is_async=is_async,
                )
            except Exception:
                continue
        return tools

    def _write(self, obj: JSON):
        data = json.dumps(obj, ensure_ascii=False)
        self.stdout.write(data + "\n")
        self.stdout.flush()

    def _error(self, id_: t.Any, code: int, message: str, data: t.Optional[JSON] = None):
        err = {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}
        if data is not None:
            err["error"]["data"] = data
        self._write(err)

    def _handle_initialize(self, id_: t.Any, params: JSON):
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": "0.1.0"},
        }
        self._write({"jsonrpc": "2.0", "id": id_, "result": result})

    def _handle_tools_list(self, id_: t.Any):
        items = []
        for tool in self._tools.values():
            items.append(
                {"name": tool.name, "description": tool.description, "inputSchema": tool.params_schema}
            )
        self._write({"jsonrpc": "2.0", "id": id_, "result": {"tools": items}})

    async def _call_tool(self, tool: ToolSpec, args: JSON):
        unexpected = set(args.keys()) - set(tool.params_schema.get("properties", {}).keys())
        if unexpected:
            raise ValueError(f"Unexpected arguments: {sorted(unexpected)}")
        sig = inspect.signature(tool.func)
        bound = sig.bind_partial(**args)
        bound.apply_defaults()
        if tool.is_async:
            return await tool.func(*bound.args, **bound.kwargs)
        return tool.func(*bound.args, **bound.kwargs)

    def _to_mcp_content(self, result: t.Any) -> list[JSON]:
        try:
            if isinstance(result, (dict, list)):
                text = json.dumps(result, ensure_ascii=False)
            else:
                text = str(result)
        except Exception as e:
            text = f"<unserializable result: {e}>"
        return [{"type": "text", "text": text}]

    async def _handle_tools_call(self, id_: t.Any, params: JSON):
        name = params.get("name")
        args = params.get("arguments") or {}
        if name not in self._tools:
            self._error(id_, -32602, f"Unknown tool '{name}'")
            return
        tool = self._tools[name]
        try:
            result = await self._call_tool(tool, args)
            content = self._to_mcp_content(result)
            self._write({"jsonrpc": "2.0", "id": id_, "result": {"content": content}})
        except Exception as e:
            self._error(id_, -32000, "Tool call failed", {"exception": repr(e)})

    async def _amain(self):
        loop = asyncio.get_running_loop()

        async def async_lines():
            while True:
                line = await loop.run_in_executor(None, self.stdin.readline)
                if not line:
                    break
                yield line

        async for line in async_lines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                self._write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue

            method = msg.get("method")
            id_ = msg.get("id")
            params = msg.get("params") or {}

            if method in ("initialized", "ping", "$/cancelRequest"):
                if id_ is not None:
                    self._write({"jsonrpc": "2.0", "id": id_, "result": None})
                continue

            if method == "initialize":
                self._handle_initialize(id_, params)
            elif method == "tools/list":
                self._handle_tools_list(id_)
            elif method == "tools/call":
                await self._handle_tools_call(id_, params)
            elif method in ("shutdown", "exit"):
                if id_ is not None:
                    self._write({"jsonrpc": "2.0", "id": id_, "result": None})
                break
            else:
                self._error(id_, -32601, f"Method not found: {method}")

    def run(self):
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
            except Exception:
                pass
        asyncio.run(self._amain())


    def main(self, name: str = "mcpkit_server", target: t.Any = None):
        srv = MCPClassServer(name, _resolve_target(target))
        srv.run()


