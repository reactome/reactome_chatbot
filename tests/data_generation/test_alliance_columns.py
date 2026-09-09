"""The Alliance metadata column lists are MITAB schemas, declared twice.

A missing comma between two adjacent string literals is not a syntax error in
Python -- it silently concatenates them. That had happened three times in
`molecular_interaction`, collapsing seven columns into three, so those metadata
fields were never populated. `genetic_interaction` is the same schema written
correctly, which makes it a usable oracle.

Note the linter cannot catch this: ruff's ISC001 flags implicit concatenation on
one line, but it is disabled because it conflicts with the formatter -- and the
formatter *joins* such literals, destroying the evidence. Hence a test.
"""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[2] / "src/data_generation/alliance/__init__.py"
)


def _column_lists() -> dict[str, list[str]]:
    tree = ast.parse(SOURCE.read_text())
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and isinstance(value, ast.List):
                # `.value` on an ast.Constant is str | bytes | int | ... , not
                # str. A newer typeshed says so, and " ".join below would have
                # raised on a list holding anything else.
                items = [
                    e.value
                    for e in value.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                ]
                if items and "interactor" in " ".join(items):
                    found.setdefault(str(key.value), items)
    return found


def test_the_two_mitab_schemas_match() -> None:
    lists = _column_lists()
    molecular = lists["molecular_interaction"]
    genetic = lists["genetic_interaction"]
    assert set(molecular) == set(genetic), (
        "the two MITAB column lists have diverged; a missing comma silently "
        "concatenates adjacent entries"
    )
    assert len(molecular) == len(genetic)


def test_no_column_name_starts_with_another_column_name() -> None:
    """Catches the concatenation directly, for lists with no second copy.

    Tests the prefix rather than a substring: "Alt. ID(s) interactor A" legitimately
    *contains* "ID(s) interactor A", but a concatenation always begins with the
    column that swallowed the comma.
    """
    for name, columns in _column_lists().items():
        for column in columns:
            prefixes = [
                other
                for other in columns
                if other != column and column.startswith(other) and len(other) > 8
            ]
            assert not prefixes, (
                f"{name}: {column!r} begins with {prefixes} -- "
                "likely a missing comma between two adjacent string literals"
            )
