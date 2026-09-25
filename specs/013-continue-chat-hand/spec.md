# Feature Specification: Continue in chat

**Feature Branch**: `013-continue-chat-hand`

**Created**: 2026-09-25

**Status**: Stories 1 and 2 built and verified end to end; Story 3 needs a shared store

**Input**: Adam, 2026-09-25: "with the chat on both the search page and the analysis results we want there to be a button to go to the [chat] interface … two options … either the logged in version or the guest version of the chat app and have the context already be set with either the search results or the analysis summary with the summary data." And: "the chat should open in a new tab."

## Context

The website already shows a chatbot-written summary in two places, both
served by this repository:

| where | endpoint | cached? |
|---|---|---|
| search page | `POST /api/answer` (spec 010) | no |
| analysis results | `POST /api/analysis/summary` (spec 011) | yes — `SummaryStore`, keyed `(token, release, tier)` |

The feature is a **"Continue in chat"** button beside each summary. It opens
the chat app in a new tab, guest or logged in, with the conversation already
holding what the user was looking at, so their first message can be a
follow-up rather than a restatement.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Continue an analysis summary in the guest chat (Priority: P1)

A user reads the summary of their enrichment analysis, clicks "Continue in
chat (guest)", and a new tab opens on the chat with that summary already in
the thread. They ask "which of these pathways involve TP53?" and get an
answer grounded in their analysis.

**Why this priority**: analysis summaries are already cached and carry a
disclosure tier, so the handoff has everything it needs; and the guest chat
is the one path verifiable on beta.

**Independent test**: from a real analysis token, the new tab's thread begins
with the same summary text the website showed, and a follow-up question is
answered using the analysis.

### User Story 2 — Continue a search-page answer (Priority: P2)

The same, from the search page's AI answer and its results.

**Why P2**: `/api/answer` does not keep what it generated, so it needs a
store before the handoff can show the *same* answer (see FR-002).

### User Story 3 — Continue in the logged-in chat (Priority: P2)

The same handoff into `/chat/personal/`, so the conversation is kept in the
user's history.

**Why P2, and why it cannot be verified on beta**: `/chat/personal` is not
deployed on beta — it needs a second container, Postgres, and OAuth whose
redirect URIs are bound to reactome.org. Verifiable in production only.

## Requirements *(mandatory)*

- **FR-001 Pass a reference, never the content.** No search result, summary
  text or analysis data in a URL. URLs are written to nginx logs, leak to
  other sites through `Referer`, and have length limits. The website asks the
  chatbot for a short-lived handoff ID; the link carries only that ID.

- **FR-002 The chat continues the summary the user saw, and does not
  regenerate it.** Answers are not reproducible here — the same question on
  the same build scores ~0.33 similarity run to run — so a regenerated summary
  would greet the user with different text from the one they clicked from.
  The handoff hands over the stored text. For analysis summaries it already
  exists in `SummaryStore`; for search answers a store is required first.

- **FR-003 The disclosure choice carries over.** A summary produced at the
  `aggregate` tier continues at `aggregate`: the chat model receives the same
  allow-listed view the summary was built from, and nothing wider. Continuing
  in chat must not silently send OpenAI more than the user agreed to on the
  website. The `identifiers` tier carries over only if it was chosen there.

- **FR-004 The handoff ID travels in the URL fragment**
  (`/chat/guest/#handoff=<id>`). Browsers never send the fragment to a
  server, so the ID does not reach nginx logs or `Referer` at all — FR-001
  enforced by the browser rather than by us remembering.

- **FR-005 The handoff is bound to the tab, not the browser.** The link opens
  in a new tab (FR-008), so a user can open several. A cookie is shared by
  every tab: two handoffs in quick succession could overwrite each other
  before the first tab connects, and that tab would open on the *wrong*
  context. The ID is instead read by a script in the tab itself and sent over
  that tab's own connection (FR-006).

- **FR-006 Mechanism (to be prototyped).** Chainlit 2.11 does not give the app
  the page URL on connect — its handler reads cookies only. It does offer
  `custom_js` (a script injected into the chat page) and `@cl.on_window_message`
  (a server hook for messages posted in the page). The script reads the
  fragment and posts the ID; the hook redeems it and seeds the thread. The
  risk to prove first: the post must arrive after the tab's socket is
  connected, or it is lost. The script retries until the server acknowledges.

- **FR-007 Valid for minutes, not once.** A single-use ID would give an empty
  chat on a reload. The ID is redeemable for a short window (target: 15
  minutes) and then refused. It grants read access to one summary, so the
  window is short and IDs are unguessable (≥128 random bits).

- **FR-008 New tab.** The website opens the chat with `target="_blank"` and
  `rel="noopener noreferrer"`, so the chat tab cannot reach back into the
  website tab and receives no `Referer`.

- **FR-009 An expired or unknown ID says so.** The chat opens normally and
  tells the user the context could not be loaded, rather than silently
  starting an empty conversation they will assume has the context.

## Who builds what

| | repo |
|---|---|
| the two buttons, the new-tab link, requesting a handoff ID | WebsiteAngular |
| `POST /api/handoff`, the store, the fragment script, the hook that seeds the thread | reactome_chatbot |

## Out of scope

- Seeding the chat with the raw search results or the full analysis result.
  The chat receives what the summary was built from, bounded as it already
  is, and can fetch more through its own tools.
- Carrying a handoff across devices or accounts.

## Open questions

- Whether the button should be two buttons (guest / logged in, as asked) or
  one that the chat resolves after login. Two is what was asked for; recorded
  so the choice is visible, not reopened.
- The handoff window length (FR-007) — 15 minutes is a guess to be revisited
  once real use shows how long people take.

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
on "couldn't load the summary". Two ways to do Story 3, to decide later:

- a store both processes share (production already has Postgres for the
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
