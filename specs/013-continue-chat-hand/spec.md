# Feature Specification: Continue in chat

**Feature Branch**: `013-continue-chat-hand`

**Created**: 2026-09-25

**Status**: Historical record of the Continue-in-chat work, 2026-09-25 (c036dd6, a8cda58, 2eac8ed), with review fixes on 2026-10-03. Stories 1 and 2 (an analysis summary or a search-page answer, into the chat served by the same process) are built, and were verified end to end on 2026-09-25. Story 3 (into the logged-in chat) is not built. The requirements below were set for that work, not for later changes. Current behaviour is in `src/api/handoff.py`, `src/handoff/`, `public/custom.js:95-146` and `bin/chat-chainlit.py:272-319, 502-543`.

**Input**: Request, 2026-09-25: "with the chat on both the search page and the analysis results we want there to be a button to go to the [chat] interface … two options … either the logged in version or the guest version of the chat app and have the context already be set with either the search results or the analysis summary with the summary data." And: "the chat should open in a new tab."

## Context (2026-09-25)

The website shows a chatbot-written summary in two places, both served by this
repository:

| where | endpoint | kept? |
|---|---|---|
| search page | `POST /api/answer` (spec 010) | Not when this spec was written. Since 2eac8ed (2026-09-25) each answered stream is kept for one hour under an `answer_id`, for handoff only; answers are never reused (`src/api/answer_store.py:27`) |
| analysis results | `POST /api/analysis-summary` (spec 011) | yes — `SummaryStore`, keyed `(token, release, tier)` (`src/analysis/store.py:46-54`) |

The feature is a **"Continue in chat"** button beside each summary, opening
the chat app in a new tab with the conversation already holding what the user
was looking at, so their first message can be a follow-up rather than a
restatement. Guest and logged-in versions were requested; only the chat served
by the process that minted the handoff can be opened (Story 3).

## User stories (set 2026-09-25)

Story numbers are kept because code and tests cite them
(`src/handoff/store.py:59`, `src/api/answer_store.py:3`,
`tests/api/test_handoff_endpoint.py:249`, `tests/api/test_answer_endpoint.py:547`).

### User Story 1 — Continue an analysis summary in the guest chat (P1; built 2026-09-25, c036dd6)

A user reads the summary of their enrichment analysis, clicks "Continue in
chat (guest)", and a new tab opens on the chat with that summary already in
the thread. They ask "which of these pathways involve TP53?" and get an
answer grounded in their analysis.

**Why P1 at the time**: analysis summaries were already cached and carried a
disclosure tier, so the handoff had everything it needed; and the guest chat
was the one path verifiable on beta.

**Check used**: from a real analysis token, the new tab's thread begins with
the same summary text the website showed, and a follow-up question is
answered using the analysis. Result under "What was built" below.

### User Story 2 — Continue a search-page answer (P2; built 2026-09-25, 2eac8ed)

The same, from the search page's AI answer and its results.

**Why P2 at the time**: `/api/answer` did not keep what it generated, so it
needed a store before the handoff could show the *same* answer (FR-002). That
store is `src/api/answer_store.py`.

### User Story 3 — Continue in the logged-in chat (proposed P2; not built)

Proposed: the same handoff into `/chat/personal/`, so the conversation is kept
in the user's history.

**Status**: not built. It is blocked by in-process state: the handoff store
and the summary cache live in each process's memory
(`src/handoff/store.py:22-24`, `src/api/analysis_summary.py:55`), and the guest
and logged-in chats are separate processes, so a handoff minted by one cannot
be claimed by the other. `POST /api/handoff` therefore returns only the
minting process's own chat path (`src/api/handoff.py:93-95, 169-181`). The two
ways forward recorded on 2026-09-25 are under "What was built" below; no
choice between them is recorded, nor whether Story 3 is still wanted.

As recorded on 2026-09-25, it could not be verified on beta: `/chat/personal`
was not deployed there, needing a second container, Postgres, and OAuth whose
redirect URIs are bound to reactome.org.

## Requirements of this work, and where they stand (checked 2026-10-21)

| ID | Required by the work (2026-09-25) | Today |
|---|---|---|
| FR-001 | Pass a reference, never the content: no search result, summary text or analysis data in a URL, because URLs reach nginx logs, leak through `Referer` and have length limits. The website asks for a short-lived handoff ID and the link carries only that | Holds: `POST /api/handoff` returns `id`, `expires_in` and `path` (`src/api/handoff.py:168-182`); the content stays in the server-side store |
| FR-002 | Continue the summary the user saw, from stored text, not regenerated: the same question on the same build scores ~0.33 similarity run to run (measured 2026-09-18 for spec 010), so a regenerated summary would differ from the one clicked from | Holds. An analysis handoff copies the text stored for that token, release and tier, and is refused if none is stored (`src/api/handoff.py:149-166`); a search handoff copies the answer kept under `answer_id` (`handoff.py:129-140`). The copy is taken at mint (`src/handoff/store.py:3-14`), and the reader sees it verbatim (`src/handoff/seed.py:139-155`) |
| FR-003 | The disclosure tier carries over: an `aggregate` summary continues at `aggregate`, and the chat model receives the same allow-listed view the summary was built from, nothing wider; `identifiers` only if chosen on the website | Holds. The tier is part of the summary lookup, so a tier the reader did not choose finds nothing (`src/api/handoff.py:149-157`); it is stored on the handoff (`src/handoff/store.py:49-51`), and the data is rebuilt at that tier with the summary endpoint's own functions (`src/handoff/seed.py:64-82`). Search handoffs carry no tier (public pathway text). Web search is skipped on a seeded analysis thread (see "Later changes") |
| FR-004 | The handoff ID travels in the URL fragment (`/chat/guest/#handoff=<id>`), which browsers never send to a server | Holds: the path is `<CHAINLIT_URI>/#handoff=<id>` for the minting process's own chat (`src/api/handoff.py:93-95, 180`), read by `public/custom.js:126-128`. Added 2026-09-25 (a8cda58): a first-time visitor's captcha form POST dropped the fragment, so the captcha page stashes it in `sessionStorage` and `custom.js` restores it (`bin/chat-fastapi.py:246-257`, `public/custom.js:109-124`) |
| FR-005 | The handoff is bound to the tab, not the browser: a cookie is shared by every tab and two quick handoffs could seed the wrong one, so a script in the tab reads the ID and sends it over that tab's own connection | Holds (`src/handoff/window.py:7-13`, `public/custom.js:126-145`) |
| FR-006 | Mechanism, to be prototyped: Chainlit 2.11 gives the app no page URL on connect, so `custom_js` reads the fragment and posts the ID, and `@cl.on_window_message` redeems it and seeds the thread; the script retries until acknowledged, because a post before the socket connects is lost | Built as described. `custom_js` (`.chainlit/config.toml:94`) posts every 500 ms, up to 40 times (about 20 s), until acknowledged (`public/custom.js:130-145`); the hook claims and acknowledges (`bin/chat-chainlit.py:502-543`). Chainlit is still 2.11.0 (`poetry.lock`). Measurements under "What was built" |
| FR-007 | Valid for minutes, not once: a single-use ID would give an empty chat on reload. Redeemable for a short window (target 15 minutes), then refused; IDs unguessable (at least 128 random bits) | Holds: 15 minutes (`src/handoff/store.py:37`), claimable repeatedly until expiry (`store.py:96-104`), 192-bit IDs (`store.py:75-77`). The store holds at most 2,048 handoffs and is lost on restart (`store.py:22-24, 40`) |
| FR-008 | New tab: the website opens the chat with `target="_blank"` and `rel="noopener noreferrer"`, so the chat tab cannot reach back and receives no `Referer` | Website code; not verifiable from this repository |
| FR-009 | An expired or unknown ID says so, rather than starting an empty conversation the user will assume has the context | Holds: `seed.UNAVAILABLE` (`src/handoff/seed.py:158-162`) is sent when the handoff is missing or seeding fails (`bin/chat-chainlit.py:281-285, 307-309`) |

## Who was to build what (2026-09-25)

This repository's part is built (see the requirements table). The website's
part is not visible from here.

| | repo |
|---|---|
| the two buttons, the new-tab link, requesting a handoff ID | WebsiteAngular |
| `POST /api/handoff`, the store, the fragment script, the hook that seeds the thread | reactome_chatbot |

## Out of scope for this work (set 2026-09-25)

- Seeding the chat with the raw search results or the full analysis result.
  The chat receives what the summary was built from, bounded as it already
  was (`src/handoff/seed.py:64-136`). Its tools can add pathway detail, but none
  fetches the analysis by token (`src/reactome_mcp/tools.py:54-97`).
- Carrying a handoff across devices or accounts.

## Open questions (recorded 2026-09-25)

- Whether the website shows two buttons (guest / logged in, as requested) or
  one that the chat resolves after login. Two were requested. With Story 3
  unbuilt, a logged-in button from the guest deployment cannot work; what the
  website shows is not visible from this repository.
- The handoff window length (FR-007). 15 minutes was a guess, to be revisited
  once real use showed how long people take. It is unchanged
  (`src/handoff/store.py:37`), and no measurement of real use is recorded.

---

## What was built, 2026-09-25

**Story 1 (analysis summary → guest chat) works end to end**, verified in a
headless browser against a real analysis on beta: the summary comes from the
real endpoint, the handoff is minted for it, the tab opens on *the same text*
the website was given, and a follow-up ("which pathway has the lowest FDR?")
is answered from the analysis -- naming its real top pathway with its exact
FDR. The control, the same question in a chat with no handoff, cannot answer
it at all. An identifiers-tier handoff is refused when the reader chose
aggregate, and an unknown ID says so.

**The transport, measured.** A single `postMessage` in the first second after
load is lost 15 times out of 15 -- Chainlit drops window messages until its
socket is up -- and from two seconds on is claimed 6 of 6. `custom.js`
retries every 500 ms for up to 20 s. Two tabs opened together in one browser
each claimed only their own ID.

**Story 3 needs a shared store before it can work.** The handoff store, like
the summary cache it copies from, is in process memory, and the guest and
logged-in chats are separate processes. A handoff minted by one cannot be
claimed by the other. The endpoint therefore returns only `path`, for the
chat served by the process that minted it; the first version also returned a
`personal_path` from the guest deployment, which could only ever have opened
on "couldn't load the summary". Two ways to do Story 3 were recorded; no choice between them is recorded:

- a store both processes share (production already had Postgres for the
  logged-in chat's LangGraph checkpoints), with the summary cache moved into
  it too, so a handoff can cross processes and still continue the *same*
  summary; or
- the website calls the logged-in deployment's own summary and handoff
  endpoints -- simpler, but the summary would be generated a second time in
  that process, and would not be the one the reader saw (FR-002).

## Story 2, 2026-09-25

**Search-page answers can be continued.** `/api/answer` now keeps each answered
stream under an `answer_id`, emitted in `done` only when `state` is
`answered`, and `POST /api/handoff` accepts `{"kind": "search", "answer_id"}`.
Keyed per answer, never per question: two readers of one search get different
answers, and neither may continue the other's. What is kept is what the page
was sent -- after anchor and sources stripping -- not the raw model output.

**No human-presence claim for a search handoff**, unlike an analysis one.
`/api/answer` does not require it either (public pathway text), so the search
page may have none to send. Pinned in both directions: requiring it for search
fails one test, dropping it for analysis fails two.

Verified end to end through the real Turnstile gate as a first-time visitor:
a real answer, a handoff minted with no human claim, the tab opens on the
question and the same answer, and "which protein kinase were we just
discussing?" is answered "CDK5". The control, with no handoff, cannot say.

## Later changes (2026-10-03 review)

- **One claim per session** (5b28d55). Any page can post into the
  window-message channel, so a session redeems only its first handoff ID;
  later IDs are acknowledged and ignored (`bin/chat-chainlit.py:516-543`).
- **Messages wait for the seed** (5f5a7c0). A message sent while a handoff is
  being seeded waits up to 45 seconds for it, so the seed is not lost beside a
  running turn (`bin/chat-chainlit.py:617-629, 672`).
- **What the model is told** (d153fbf). The seeded analysis turn carries the
  summary endpoint's data rules, and says whether the data could not be
  fetched or is gone (`src/handoff/seed.py:64-136`).
- **No web search on an analysis thread** (1b8c939; read from the thread
  itself since 14ca5cd). The rephrased question sent to web search is built
  from history that holds the reader's analysis, so post-processing is skipped
  on a seeded thread (`bin/chat-chainlit.py:561-566`, `src/agent/graph.py:425`).
- **Escaping** (1b8c939, 5f5a7c0). The question and summary shown to the
  reader are escaped or rendered inert (`src/handoff/seed.py:139-155`).
