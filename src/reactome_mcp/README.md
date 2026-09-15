# Reactome MCP client

Talks to [`reactome-mcp`](https://github.com/reactome/reactome-mcp), which
exposes Reactome's Content and Analysis services as MCP tools.

Harvested from #127 and #137 by @GovindhKishore.

## Why the package is called `reactome_mcp`

The official MCP Python SDK is published on PyPI as `mcp`. A local package of
that name shadows it, and the failure appears only the day someone adds the
dependency — as an import that resolves to the wrong thing rather than an error.

## What is here

| | |
|---|---|
| `process.py` | starts and stops the server subprocess |
| `client.py` | JSON-RPC over its stdio |
| `tools.py` | five of the server's 53 tools, as LangChain tools |
| `probe.py` | `./bin/mcp-probe` — check the integration end to end |

Nothing here is wired into the chat graph yet. Routing questions to these tools
is `specs/007-answer-cascade`, and it belongs on the existing
`intent_classifier` rather than in a second one.

## Checking it works

```bash
export REACTOME_MCP_SERVER=~/git/reactome-mcp/dist/index.js
./bin/mcp-probe
```

```
  connected to reactome 1.4.0
  server exposes 53 tools
  chatbot wraps 5 of them
  reactome_database_info             ok   0.15s      64 chars
  reactome_species                   ok   0.39s    1988 chars
  reactome_search                    ok   0.13s    3974 chars
  reactome_get_pathway               ok   0.05s    2425 chars
  reactome_analyze_identifiers       ok   0.04s    2667 chars
```

Several things fail identically from inside the chatbot — wrong path, no node,
stale build, Content Service down, handshake rejected — and a question that
quietly falls back to retrieval reports none of them. This separates them, and
exits non-zero, so it can gate a deployment.

## Five tools, not fifty-three

Every tool description is spent from the model's context before it answers
anything, and a model choosing between 53 similarly-named tools chooses worse
than one choosing between five. The five cover what the bundle cannot do: live
search, live pathway lookup, enrichment analysis, and the two metadata
questions a snapshot cannot answer.

Add to the list when a question is being answered wrongly without the tool —
not because the tool exists.

## Transport

Two, and which one you can use depends on where this runs.

| variable | transport | where |
|---|---|---|
| `REACTOME_MCP_URL` | Streamable HTTP | **anywhere**, including the container |
| `REACTOME_MCP_SERVER` | stdio, spawning `node` | a developer's machine only |

`REACTOME_MCP_URL` wins when both are set.

**stdio cannot work in the deployed container.** The image is Python: it has no
`node`, and it does not mount reactome-mcp. `REACTOME_MCP_SERVER` can never be
satisfied there, so the live destination worked on every machine it was tested
on and none that it ships to. That is why the HTTP transport exists.

To run one alongside the chatbot:

```bash
cd ~/git/reactome-mcp && npm ci && npm run build
MCP_HTTP_PORT=4320 node dist/http-server.js
# then, for the chatbot:
REACTOME_MCP_URL=http://127.0.0.1:4320
```

reactome-mcp binds loopback by default; see `specs/002-transport-and-hosting`
in that repository for why, and for the shape of a hosted deployment.

## Transport internals

stdio, by spawning the server. reactome-mcp also serves Streamable HTTP, so a
hosted instance can be used instead once there is one; that is
`specs/006-mcp-hosting`. Keeping the transport inside `MCPProcessManager` is
what makes that swap small.

## Three things the original client did not do

**The `initialize` handshake.** It went straight to `tools/call`. The server
accepts that today because the SDK is lenient — verified — but the protocol
requires it, and relying on leniency means the day an SDK release enforces it,
every call fails at once. Doing it properly also replaced an arbitrary
`sleep(1)` used to decide the server had started: a successful initialize *is*
the readiness check.

**Matching replies to requests.** It returned the next line on stdout, whatever
it was. A notification arriving between request and reply would be read as the
answer, and every later call would be one reply out of step — answering each
question with the previous question's answer. Nothing raises; it just returns
the wrong thing, plausibly. Two tests pin this.

**One call at a time.** A lock serialises each write/read pair, because two
coroutines interleaving on one pipe is the same desync by another route.
