# Running the chatbot as Plant Reactome

`plantreactomedev.gramene.org` serves `/chat/personal` from this repository's
image. It is **the same image Reactome runs** — nothing is forked, no brand
string is compiled in. What differs is four things, and all four are
configuration:

| | Reactome | Plant Reactome |
|---|---|---|
| `config.yml` → `profiles:` | `React-to-Me` | `Plant Reactome` |
| brand assets in `/app/public` | `public/` | `public-plantreactome/`, via `BRAND_DIR` |
| welcome screen | `chainlit.md` | `chainlit.plantreactome.md`, mounted over it |
| env file | `.env` | `.env` (see §4) |

`src/util/branding.py` is the one place a deployment says who it is. It reads
`profiles:` from `config.yml` and everything else — landing page, captcha error
pages, the login page's title, the message avatar filename — follows from it.
Reactome's branding is the default *in code*, so a checkout that sets nothing
serves exactly what it serves today.

## 1. Run

From the repository root:

```bash
docker compose -f compose.yaml -f deploy/plantreactome/compose.plantreactome.yaml up -d
```

`.env` in the repository root is read automatically for `${...}` interpolation,
so no `--env-file` is needed. Relative paths inside the override resolve against
the **first** `-f` file's directory — the repository root — which is why
`./deploy/plantreactome/secrets/empty` and `${BRAND_DIR:-./public}/favicon.svg`
work from here.

The container listens on **8002**, not 8000. `ports: !override` in the override
file replaces upstream's mapping rather than adding to it.

First start takes a while. `bin/chat-fastapi.py`'s lifespan loads the
caller-token key and builds the graph — 50 to 90 seconds — before uvicorn
accepts a connection. `docker logs` prints `Agent graph ready in %.1fs` when
it's done; until then, requests hang rather than failing.

## 2. Why this file is here, and not named `compose.override.yaml`

`compose.override.yaml` **in the project directory** is Compose's conventional
override name: a bare `docker compose up` loads it with no `-f` and no warning.
This deployment's override changes the published port, replaces all seven
secrets with local file paths, parks `vault` and `pgadmin` behind profiles, and
bind-mounts `./src` and `./bin` over the image. On Reactome's host that is not a
misconfiguration, it is a broken `up`.

So it lives here, under a name Compose will not discover. It loads only when
named. Keep it that way — renaming it to `compose.override.yaml` and moving it
to the repository root would make this branch silently rewire any deployment
that checks it out.

`!override` inside the file is load-bearing, not decoration. A top-level
`secrets:` is a mapping and mappings **merge** on override, so re-declaring a
secret with a plain `file:` would leave upstream's `external: true` beside it —
and a secret with both is invalid.

## 3. The env file

`.env` and `.env.*` are gitignored. Two have been used on this codebase and
**they are not the same file**:

| | `.env` | `.env.dev` |
|---|---|---|
| `CHAINLIT_URL` | `http://localhost:8002` | `https://plantreactomedev.gramene.org` |
| Cloudflare keys | Cloudflare's dummy test pair | real |

`.env` is the one Compose reads. A local run therefore exercises neither the
real Turnstile keys nor the real `CHAINLIT_URL`, which is fine for the chat and
misleading for anything involving a redirect or a captcha.

**An env file only feeds `${...}` interpolation. It never becomes the
container's environment.** A variable reaches the container only if
`compose.yaml` — or this override — names it in an `environment:` block. That
single rule has produced three separate bugs here, all silent:

- `CALLER_TOKEN_PUBLIC_KEY_PATH` set in the env file, never forwarded. Harmless
  in this case, because `compose.yaml` hardcodes the path anyway (§7).
- `OPENAI_BASE_URL` and all four `LLM_*` variables never forwarded. The chat
  model fell back to the SDK's default `https://api.openai.com/v1` carrying the
  copilot-dev key, and answered `401 ... You can find your API key at
  platform.openai.com`. Fixed by the `environment:` additions in this override.
- `CHAT_ENV` and `PYTHON_PATH` were in the env file and forwarded by nothing --
  `PYTHON_PATH` misspelled besides (`PYTHONPATH`, already `/app/src` in the
  image). Both have since been removed. A variable that appears in an env file
  but in no `environment:` block is documentation for a setting that does not
  exist, and it reads exactly like one that does.

When you add a variable, add it to `chainlit.environment` in this file or it
does not exist as far as the application is concerned.

## 4. Model and embedding endpoints: two pairs, and both halves must arrive

| Purpose | Base URL | Key | Notes |
|---|---|---|---|
| Chat model | `LLM_BASE_URL` | `LLM_API_KEY` | `LLM_MODEL`, and `LLM_API_KEY_HEADER=api-key` — Azure APIM reads its subscription key from that header and answers the SDK's `Authorization: Bearer` with 401 |
| Embeddings | `OPENAI_BASE_URL` | `OPENAI_API_KEY` | OpenAI-compatible endpoint serving **bge-m3** |

The embedding model is **not configurable** — `agent.graph.resolve_embedding_model()`
derives it from the installed bundle, because a query embedded with a different
model than the stored vectors returns nonsense rather than an error. Check the
active bundle with `cat embeddings/current`; here it is
`openai/bge-m3/plantreactome/Release68`. Setting `EMBEDDING_MODEL` to something
else logs an error and still uses the configured value, so don't.

**Bundles do not come from S3 here.** They are built elsewhere, shipped between
hosts as tarballs, and extracted into `./embeddings/`, which `compose.yaml`
bind-mounts read-write at `/app/embeddings`. `embeddings/current` is a one-line
pointer to the active bundle and `resolve_embedding_model()` reads it, which is
why the model is a fact about the installed data rather than a setting that can
be gotten wrong.

`./bin/embeddings_manager` does exist, and it does install from S3 -- but the
bundles it publishes are Reactome's `openai/text-embedding-3-large/*`, built
with a different model than the Plant Reactome vectors. Installing one here
would make retrieval meaningless rather than fail loudly.

`config.yml` sets `llm.provider: openai-responses` because the APIM gateway
publishes only `POST /openai/responses`; `chat/completions` gets a gateway-level
404 that no value of `LLM_MODEL` can fix.

## 5. Brand assets

Chainlit resolves every brand asset by **fixed filename in one flat directory**
(`chainlit/server.py`). There is no brand directory to point it at, only names
to replace, which is why the override mounts five individual paths:

| Slot | Chainlit's lookup |
|---|---|
| `/logo?theme=light` | glob `public/logo_light.*` |
| `/logo?theme=dark` | glob `public/logo_dark.*` |
| `/favicon` | glob `public/favicon.*` |
| `theme.json` | read server-side into the page |
| `/avatars/{id}` | `public/avatars/{slug}.*`, falling back to the favicon |

The avatar slug is `config.ui.name.strip().lower().replace(" ", "_").replace(".", "_")`,
and `config.ui.name` is set from `BRANDING.chatbot_name` in
`bin/chat-chainlit.py`. So `Plant Reactome` requires
`public-plantreactome/avatars/plant_reactome.jpg`, and a miss serves the favicon
instead of failing. `public/avatars/React-to-Me.jpg` is unreachable for the same
reason — the glob is case-sensitive and the slug is lowercased.

`BRAND_DIR` selects the tree and **must be set in the env file**. It is
`${BRAND_DIR:-./public}`, so unset *and empty* both mean Reactome, and Compose
emits no warning. A Plant Reactome host with `BRAND_DIR` missing serves Reactome's
logo from a Plant Reactome page and logs nothing.

Two things `theme.json` does **not** do:

- It carries no `Cache-Control`. Chainlit returns assets via `FileResponse`
  without `max_age`, so the browser picks its own lifetime from `Last-Modified`
  and can serve a logo for weeks without a single request. See §8.
- Its `--primary` reaches the Tailwind/shadcn surfaces — buttons, toggles,
  badges, toasts, the chat UI. The login card's provider buttons did not respond
  to it. Overriding those needs CSS, and `custom_js`/`custom_css` is the hook.

`custom.js` and `elements/` exist in `public-plantreactome/` but are **not
mounted** — the five slots above are the whole list. Editing them does nothing
until they join the mount list.

## 6. Docker secrets

`compose.yaml` declares all seven secrets `external: true`, which is Swarm-only:
on a single engine `docker secret create` answers *"This node is not a swarm
manager"*. This override replaces the block with `file:` sources.

Six of the seven point at `deploy/plantreactome/secrets/empty`, a committed
**zero-byte** file. That is not a placeholder for a value you forgot to fill in.
`compose.yaml`'s own `environment:` block already supplies those six, and
`get_secret()` in `src/util/secrets.py` returns the mounted file's contents *or*
falls back to `os.getenv(name)` — so an empty file is exactly equivalent to not
mounting one. It is mounted anyway to keep the shape upstream expects.

It must be **empty, not blank**: `mounted_secrets()` stats file size rather than
reading contents, so a file holding a single newline is logged as a supplied
secret.

`mode:` is deliberately absent from the secret definitions. Docker Compose
ignores `mode`/`uid`/`gid` for file-sourced secrets — they are bind mounts, so
the host file's own permissions are what the container sees, and the image runs
as `appuser`. Write the real key 0644 on the host; `bin/make-caller-token-keypair.py`
does it.

To move a value out of the env file and into a secret, write it into its own
file and repoint that entry. A mounted file beats the environment —
`bin/chat-chainlit.py` calls `load_secrets_to_environ(SECRET_NAMES)` at import,
before anything reads `os.environ` — which also makes the container immune to
whatever the shell running `docker compose` happens to have exported.

## 7. The caller-token verifying key

```bash
./bin/make-caller-token-keypair.py deploy/plantreactome/secrets
shred -u deploy/plantreactome/secrets/caller_token_private.pem
```

The answer endpoint verifies a token the website mints per request, and the app
**refuses to start** without the key — deliberately, because an endpoint that
accepts everything because its key is missing is the worst available outcome and
would test clean. Chainlit is mounted on the same app, so a missing key takes
`/chat` down with it.

**The secret's name fixes the path, not the file's.** `compose.yaml` hardcodes
`CALLER_TOKEN_PUBLIC_KEY_PATH=/run/secrets/CALLER_TOKEN_PUBLIC_KEY`, and a secret
is mounted at `/run/secrets/<its name in the top-level secrets: block>`. The
script's closing hint prints `CALLER_TOKEN_PUBLIC_KEY_PATH=/run/secrets/caller_token_public.pem`,
which is correct for the beta `docker run --env-file` path and inert here.

A self-generated pair means `/api/answer` refuses every token while the chat
looks perfectly healthy. That is the safe direction — it fails closed — but it
is not integrated. The real public half comes from whoever mints the tokens;
`deploy/beta/caller_token_private.pem` still existing suggests beta's pair is a
local test pair rather than that key.

## 8. Verifying a change actually took effect

Three things stand between an edit and what you see, and all three fail silently.

**Bind mounts pin the inode.** `cp`, `mv`, `git checkout` and vim's default
write all produce a new inode, and the mount keeps serving the old bytes
indefinitely. Recreate rather than restart:

```bash
docker compose -f compose.yaml -f deploy/plantreactome/compose.plantreactome.yaml \
  up -d --force-recreate chainlit
```

**Check the file inside the container, not the URL:**

```bash
docker exec $(docker ps -qf publish=8002) sha256sum /app/public/logo_dark.png
```

The URL is a worse probe than it looks: the captcha middleware wraps the mounted
Chainlit app, so `curl` without a pass cookie gets `307 → /chat/personal/verify_captcha_page`
with an empty body. `curl … | sha256sum` then returns `e3b0c442…`, the hash of
the empty string, which looks like an answer and isn't one.

**The browser cache is not cleared by "Clear site data."** That clears storage —
cookies, localStorage, IndexedDB, service workers — not the HTTP disk cache.
DevTools' *Empty Cache and Hard Reload* does. While iterating, keep the Network
panel open with **Disable cache** ticked; it's the only setting that survives a
whole session.

To confirm a theme value reached the page at all, use computed style on the
element rather than guessing at an injection variable:

```js
getComputedStyle(document.documentElement).getPropertyValue('--primary')
```

## 9. Postgres

`compose.yaml`'s initdb creates databases named `chainlit` and `langgraph` with
no tables, while `POSTGRES_CHAINLIT_DB` and `POSTGRES_LANGGRAPH_DB` default to
`chatbio_chainlit` and `chatbio_langgraph`, and Chainlit's SQLAlchemy layer does
not create its own schema. So the repository's `./initdb/` is mounted over it —
same container path, last file wins. This is worth reporting upstream: as
written, `compose.yaml` cannot start a working database for any deployment.

`postgres` runs with `--auth-host reject`, so there is no TCP path to it at all.
Everything connects over the `postgres-socket` volume. `pgadmin` therefore
cannot connect over the network even when its profile is enabled.

If `./data` holds a Postgres **13** cluster, do not point a 17 container at it —
17 cannot open a 13 data directory. Dump with `pg_dumpall` first if the history
matters.

## 10. Parked services

`chainlit-guest`, `vault` and `pgadmin` are behind profiles and do not start.

- `chainlit-guest` (formerly `chainlit-no-login`) serves no surface here — the
  site's Apache config routes all of `/chat/` to the authenticated container.
- `vault` is parked, but note the trap: `get_db_uri()` takes the Vault path the
  moment `/tokens/postgres-app/token` exists, waits five minutes for an unsealed
  Vault, and then raises **at import**. If that named volume was ever populated,
  `docker volume rm reactome_chatbot_vault-postgres-app-token`.
- `pgadmin` needs `--profile admin` and a socket mount to be useful at all.

## 11. Known gaps

- `favicon.svg` and `avatars/plant_reactome.jpg` in `public-plantreactome/` are
  still Reactome artwork — byte-identical copies of the originals.
- `theme.json` still carries Reactome's blue (`195 61% 47.3%`) in four places:
  `--primary` and `--ring` in each of the light and dark blocks. The replacement
  colour should come from Gramene's own CSS, not from a colour histogram of the
  logo — `PR_logo_large_v2.png` is a pictorial illustration and yields artwork
  highlights, not a UI accent. `--primary-foreground` is white in both themes,
  which constrains how light the primary can be.
- `logo_dark.png` is a byte copy of `logo_light.png`. If the logo has dark glyphs
  it will be invisible in dark theme, which is where Chainlit serves it.
- `S3_BUCKET` and `S3_CHAINLIT_PREFIX` are deliberately **empty**. `compose.yaml`
  forwards both, and `bin/chat-chainlit.py` builds an S3 storage provider only
  when both are non-empty, so elements and uploaded files stay on local disk.
  They previously named `download.reactome.org` under `react-to-me/embeddings`,
  which would have written Plant Reactome's uploads into Reactome's public
  bucket under Reactome's own prefix. Set both to a Gramene bucket if thread
  history has to outlive the container -- and set **both**: one of the two
  silently disables the provider rather than half-enabling it.
- `CHAINLIT_IMAGE` is `…reactome-chatbot:latest`. Using a floating tag as the
  base of `Dockerfile.local` is the same failure mode as an unpinned transitive
  dependency, which this repository has already been bitten by once.
- `support_email` is `reactome@gramene.org`. `tests/test_welcome_pages.py` only
  asserts the *absence* of `help@reactome.org`, so it passes whatever is set here.
