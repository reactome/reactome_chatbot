"""Text placed into chat markdown, kept literal."""

import pytest

from util.markdown import escape, escape_directives, inert_html


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Chainlit's renderer has remark-directive: ":p25" was parsed as markup
        # and dropped, so "CDK5:p25 phosphorylates CDC25A" showed as "CDK5",
        # a break, then the rest.
        ("CDK5:p25 phosphorylates CDC25A", "CDK5\\:p25 phosphorylates CDC25A"),
        (
            "[CDK5:p25 x](https://reactome.org/content/detail/R-HSA-1)",
            "[CDK5\\:p25 x](https://reactome.org/content/detail/R-HSA-1)",
        ),
        # Left alone: not directive-shaped.
        ("see https://reactome.org/x", "see https://reactome.org/x"),
        ("at 12:30", "at 12:30"),
        ("ratio 1:2", "ratio 1:2"),
        ("Note: this", "Note: this"),
    ],
)
def test_directive_colons_are_escaped_and_nothing_else(
    text: str, expected: str
) -> None:
    assert escape_directives(text) == expected


def test_names_in_tables_and_handed_off_summaries_get_it_too() -> None:
    assert "CDK5\\:p25" in escape("CDK5:p25 complex")
    assert inert_html("**A:B** <b>") == "**A\\:B** \\<b>"
