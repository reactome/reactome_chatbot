# Running the chatbot behind beta.reactome.org/chat

First cut: **guest access only**. That needs no Google OAuth (whose redirect URIs
are registered for `reactome.org`, not beta) and no Postgres — the LangGraph
checkpointer falls back to `MemorySaver`, so conversations live in memory and are
lost on restart. Chat history and the `/chat/personal` route come later.

**Turnstile is enforced on `/chat/guest/`** since 2026-09-18, using production's
keys -- the site key now lists `beta.reactome.org` as well as `reactome.org`, so
beta no longer needs to run without one. Rotating the key in production means
rotating it here too, or beta breaks.

Leaving `CLOUDFLARE_SECRET_KEY` unset used to make the middleware bypass itself
silently. It now refuses to start instead, unless `CHAT_REQUIRES_HUMAN=0` says
deliberately that a deployment wants no human check. "There is captcha
middleware" and "the chat is gated" were previously different statements with
nothing to tell them apart.

## 1. Embeddings

The container answers nothing without a bundle. Available on S3 (probe with
`curl -I`, anonymous listing is denied):

| bundle | size |
|---|---|
| `openai/text-embedding-3-large/reactome/Release95` | 1.3 GB |
| `openai/text-embedding-3-large/reactome/Release94` | 1.4 GB |
| `openai/text-embedding-3-large/reactome/Release91` | 1.0 GB |
| `openai/text-embedding-3-large/reactome/Release90` | 2.0 GB |
| `openai/text-embedding-3-large/reactome/Release89` | 1.9 GB |

```bash
mkdir -p embeddings
docker run --rm -v "$PWD/embeddings:/app/embeddings" \
  public.ecr.aws/reactome/reactome-chatbot:e398a37 \
  ./bin/embeddings_manager install openai/text-embedding-3-large/reactome/Release95
```

Budget roughly 1.3 GB download plus 2–3 GB extracted, on top of a ~4–6 GB image.

## 2. Config

```bash
cp config_default.yml config.yml     # prod's config.yml is byte-identical to this
```

## 3. Environment

Copy `env.beta.template` to `.env.beta` and fill in the two keys. Do not reuse
prod's `CHAINLIT_URL`, `CHAINLIT_ROOT_PATH` or OAuth values — they point at
`reactome.org` and will break asset URLs and logins on beta.

## 4. The caller-token verifying key

The answer endpoint (`/chat/guest/api/answer`) verifies a token the website mints
for each call. This host holds **only the public half** and can therefore verify
but never mint -- which is the point of choosing an asymmetric algorithm, and why
no private key for this path should exist here.

The app **refuses to start** without the key. That is deliberate: an endpoint that
accepts everything because its key is missing is the worst outcome available, and
it would test clean. But Chainlit is mounted on the same app, so a missing key
takes `/chat` down with it -- which is why `update-beta-chat.sh` checks the key is
readable *before* it stops the running container.

The public half comes from the website team. Ask them for it, write it to
`deploy/beta/caller_token_public.pem` (mode 644 -- the image runs as `appuser` and
must read it), and restart. Tokens signed by the previous key stop verifying the
moment you do, which is what makes rotation a file write plus a restart.

If you need a keypair for local testing rather than the real exchange:

```bash
./bin/make-caller-token-keypair.py deploy/beta
```

It refuses to overwrite an existing key, because silently replacing one would
invalidate every token in flight with no way back. **Delete the private half once
a real key is installed**: on the verifying side it is pure liability.

## 5. Run

Bound to loopback: Apache is the only thing that should reach it.

```bash
docker run -d --name biochat_beta_guest --restart unless-stopped \
  --env-file .env.beta \
  -v "$PWD/embeddings:/app/embeddings" \
  -v "$PWD/config.yml:/app/config.yml" \
  -p 127.0.0.1:8000:8000 \
  public.ecr.aws/reactome/reactome-chatbot:e398a37

curl -s localhost:8000/chat/ | grep -o React-to-Me   # should print React-to-Me
```

The image tag matches what production runs today, so this is a like-for-like
baseline to compare against after the dependency upgrade.

Note: the landing page shows both a **Guest Access** and a **Log In** button. Only
Guest Access works in this setup; wiring Log In needs a second container on :8001
with `CHAINLIT_URI=/chat/personal`, plus Postgres and OAuth.

## 6. Apache

See `../../../WebsiteAngular/deploy/apache/install-beta-chat-proxy.sh`.

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel rewrite
sudo ~/git/WebsiteAngular/deploy/apache/install-beta-chat-proxy.sh
```

`proxy_wstunnel` is not enabled on this host today and Chainlit needs it —
without it the UI renders and then hangs with no replies.
