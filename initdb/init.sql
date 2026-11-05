-- Chainlit 2.11.0 data-layer schema.
-- Derived from the TypedDicts in the installed package (chainlit.step.StepDict,
-- chainlit.element.ElementDict, chainlit.types.ThreadDict/FeedbackDict,
-- chainlit.user.UserDict) -- NOT from docs.chainlit.io, which is missing
-- autoCollapse, icon, path, autoPlay and playerConfig.
-- The data layer builds INSERT column lists from every non-None key, so each
-- table must be a superset of its TypedDict. Run bin/check_schema.py after any
-- chainlit upgrade.

CREATE DATABASE chatbio_chainlit;
CREATE DATABASE chatbio_langgraph;
CREATE DATABASE chatbio_langgraph_no_login;

\c chatbio_chainlit

-- Literal INSERT (sql_alchemy.py:183): fixed columns, no drift risk.
CREATE TABLE users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "command" TEXT,
    "modes" JSONB,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "defaultOpen" BOOLEAN,
    "autoCollapse" BOOLEAN,
    "language" TEXT,
    "icon" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    "path" TEXT,
    "autoPlay" BOOLEAN,
    "playerConfig" JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- "threadId" is nullable on purpose: FeedbackDict has no threadId key, so a
-- NOT NULL constraint here would reject feedback built from that dict.
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
