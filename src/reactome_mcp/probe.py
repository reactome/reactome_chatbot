"""Check that the MCP server is reachable and its tools answer.

The integration has several places to go wrong that look alike from inside the
chatbot: the server path is wrong, node is missing, the build is stale, the
Content Service is down, or the handshake fails. A question that quietly falls
back to retrieval tells you none of that.

    ./bin/mcp-probe --server ~/git/reactome-mcp/dist/index.js

Exits non-zero if anything fails, so it can gate a deployment.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from reactome_mcp.client import MCPClient
from reactome_mcp.process import MCPProcessManager
from reactome_mcp.tools import create_mcp_tools

# One call per curated tool, with arguments known to return something. A tool
# that answers "0 results" is as much a failure here as one that raises: it
# means the server is up and the data is not reaching it.
CHECKS: list[tuple[str, dict[str, object], str]] = [
    ("reactome_database_info", {}, "Reactome"),
    ("reactome_species", {}, "Homo sapiens"),
    ("reactome_search", {"query": "TP53"}, "R-HSA-"),
    ("reactome_get_pathway", {"id": "R-HSA-109582"}, "Hemostasis"),
    (
        "reactome_analyze_identifiers",
        {"identifiers": ["TP53", "BRCA1", "EGFR"]},
        "Token",
    ),
]


async def probe(server_path: Path, timeout: float) -> int:
    failures = 0
    async with MCPProcessManager(server_path) as manager:
        if manager.process is None:
            print("  server did not start", file=sys.stderr)
            return 1
        client = MCPClient(manager.process, timeout=timeout)

        try:
            info = await client.initialize()
        except Exception as exc:
            stderr = await manager.stderr_tail()
            print(f"  handshake FAILED: {exc}", file=sys.stderr)
            if stderr:
                print(f"  server said:\n{stderr}", file=sys.stderr)
            return 1

        server = info.get("serverInfo", {})
        print(f"  connected to {server.get('name')} {server.get('version')}")

        tools = await client.list_tools()
        exposed = {t.get("name") for t in tools}
        print(f"  server exposes {len(tools)} tools")

        wrapped = {t.name for t in create_mcp_tools(client)}
        # The wrappers name MCP tools by string. A rename upstream would
        # otherwise surface as a confusing runtime error on a user's question.
        missing = {name for name, _args, _expect in CHECKS if name not in exposed}
        if missing:
            print(
                f"  MISSING from server: {', '.join(sorted(missing))}", file=sys.stderr
            )
            failures += len(missing)
        print(f"  chatbot wraps {len(wrapped)} of them")

        for name, args, expect in CHECKS:
            if name in missing:
                continue
            started = time.monotonic()
            try:
                text = await client.call_tool(name, args)
            except Exception as exc:
                print(f"  {name:<34} FAILED  {exc}", file=sys.stderr)
                failures += 1
                continue
            elapsed = time.monotonic() - started
            if expect not in text:
                print(
                    f"  {name:<34} answered in {elapsed:5.2f}s but did not "
                    f"contain {expect!r}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            print(f"  {name:<34} ok  {elapsed:5.2f}s  {len(text):>6} chars")

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        type=Path,
        default=os.getenv("REACTOME_MCP_SERVER"),
        help="Path to reactome-mcp's dist/index.js " "(default: $REACTOME_MCP_SERVER).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.server is None:
        raise SystemExit(
            "No MCP server path. Pass --server, or set REACTOME_MCP_SERVER to "
            "reactome-mcp's dist/index.js."
        )

    print(f"probing {args.server}")
    raise SystemExit(asyncio.run(probe(args.server, args.timeout)))
