"""The classifier must describe only the destinations that exist.

Offering one that is not wired up is worse than not having it: the model routes
there, the source is missing, and the fallback answers as though it had been
asked instead -- which is how "what tools are available on here" came to be
answered with a claim that Reactome has no analysis tool.
"""

from agent.tasks.intent_classifier import (
    SourceName,
    build_classifier_message,
    intent_classifier_message,
    resolve_active_sources,
)

TWO: frozenset[SourceName] = frozenset({"reactome", "userguide"})
THREE: frozenset[SourceName] = frozenset({"reactome", "userguide", "live"})
ONE: frozenset[SourceName] = frozenset({"reactome"})


def test_without_the_mcp_the_prompt_is_byte_for_byte_what_it_was() -> None:
    """FR-007: with the new step disabled, behaviour is exactly as before.

    Not "similar" -- identical. A deployment with no MCP server must classify
    the same questions the same way, and pay the same tokens doing it.
    """
    assert build_classifier_message(TWO) == intent_classifier_message


def test_live_is_described_only_when_it_is_available() -> None:
    assert "**live**" not in build_classifier_message(TWO)
    assert "**live**" in build_classifier_message(THREE)


def test_the_live_rule_distinguishes_content_from_scope() -> None:
    """The distinction is not "biology vs not"."""
    message = build_classifier_message(THREE)
    assert "What does CDK5 do?" in message
    assert "does Reactome have" in message


def test_userguide_only_deployments_still_get_their_block() -> None:
    message = build_classifier_message(TWO)
    assert "**userguide**" in message
    assert "**reactome**" in message


def test_an_unavailable_source_falls_back_rather_than_failing() -> None:
    # A question routed to live on a deployment without it must still be
    # answered, not dropped.
    assert resolve_active_sources("live", TWO) == ["reactome"]
    assert resolve_active_sources("live", THREE) == ["live"]
    assert resolve_active_sources("userguide", ONE) == ["reactome"]
