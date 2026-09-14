# Rebuilding the embeddings bundle

For a new Reactome release. Written 2026-09-14, ahead of Release 98, after checking
each step against this machine — the gaps below are real, not hypothetical.

## Where things stand

| | |
|---|---|
| installed bundle | `openai/text-embedding-3-large/reactome/Release95` |
| built | 2026-09-02 |
| Reactome current | **97**, with 98 nearly done |

The bundle was already two releases behind on the day it was built, so this is not
drift — nothing rebuilt it against a current release. That is the whole reason this
document exists.

## Pre-flight: three things that are not ready

Checked, not assumed. Each one fails partway through a long job rather than at the
start, which is the worst way to find out.

### 1. Disk — the tightest constraint

```
existing bundle   3.4 G
free on /         5.2 G   (95% used)
```

A second bundle needs roughly another 3.4 G, leaving under 2 G for the operating
system, Docker and the build's own scratch space. **This will not fit comfortably.**

Before starting: free space, or build somewhere else. `deploy/beta/reclaim-docker-space.sh`
exists for the Docker side. Note the old bundle cannot simply be deleted first —
the running chatbot is serving from it.

### 2. S3 — no credentials on this machine

```
$ ./bin/embeddings_manager ls-remote
botocore.errorfactory.AccessDenied: ... ListObjects ... Access Denied
```

There is no `~/.aws`, no `AWS_*` in the environment, and nothing in `.env`. So
`pull`, `push` and `ls-remote` all fail here. CI authenticates to AWS by OIDC role
assumption, but only for the ECR image push — not for the embeddings bucket.

`push` is how a new bundle reaches production, so **this must be solved before the
rebuild, not after it**.

### 3. Neo4j — needed, undocumented

`embeddings_manager make` reads the Reactome graph database and defaults to
`bolt://localhost:7687`. Nothing in `.env` or `env_template` configures it, so the
connection must be passed explicitly:

```bash
./bin/embeddings_manager make \
    openai/text-embedding-3-large/reactome/Release98 \
    --neo4j-uri bolt://<host>:7687 \
    --neo4j-username <user> \
    --neo4j-password <password>
```

## What is ready

The generation code itself. Every `data_generation` module imports cleanly on
LangChain 1.x after the 2026-09-09 upgrade — `reactome`, `neo4j_connector`,
`uniprot`, `alliance` and `userguide`. That was the part most likely to have rotted,
and it has not.

## The sequence

```bash
./bin/embeddings_manager make openai/text-embedding-3-large/reactome/Release98 \
    --neo4j-uri ... --neo4j-username ... --neo4j-password ...
./bin/embeddings_manager push openai/text-embedding-3-large/reactome/Release98
./bin/embeddings_manager use  openai/text-embedding-3-large/reactome/Release98
```

Then restart the chatbot so `AgentGraph` picks up the new bundle, and confirm with
`./bin/embeddings_manager which`.

## After rebuilding, check retrieval actually improved

A new bundle is a change to what reaches the model, so constitution Article II
applies. `bin/retrieval_baseline` captures before and after against a fixed question
set:

```bash
./bin/retrieval_baseline capture --out before-98.json    # while 95 is active
# ... rebuild and switch ...
./bin/retrieval_baseline capture --out after-98.json
./bin/retrieval_baseline compare before-98.json after-98.json
```

Expect large differences — that is the point — but the comparison shows *what*
changed rather than only that something did.

## A note on chromadb

The bundle is written by `langchain_community.vectorstores.Chroma` in
`data_generation` and read by `langchain_chroma` in the retriever. Both sit on
chromadb, which is pinned below 1.0 (see `pyproject.toml`): chromadb 1.x migrates a
bundle's sqlite in place on first open, which needs write access the container does
not have. A bundle built now is readable by both.
