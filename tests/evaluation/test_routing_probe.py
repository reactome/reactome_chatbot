"""The probe that watches for a collection nobody routes to.

Its whole purpose is the guard `answer-sweep` cannot provide, so the thing to
test is that the guard fires -- not that the happy path is quiet.
"""

from evaluation.routing_probe import PROBES, Probe, Result, report
from retrievers.reactome.metadata_info import reactome_descriptions_info


def _result(expect: str, chose: list[str]) -> Result:
    probe = Probe(question=f"about {expect}", expect=expect, why="test")
    return Result(probe=probe, chose=sorted(chose))


def _all_collections_covered() -> list[Result]:
    return [_result(name, [name]) for name in reactome_descriptions_info]


def test_every_collection_has_at_least_one_probe() -> None:
    # A collection with no probe cannot be reported as never chosen, which
    # would make this file quietly useless for exactly the collection it was
    # written for.
    for name in reactome_descriptions_info:
        assert any(p.expect == name for p in PROBES), f"nothing probes {name}"


def test_two_probes_per_collection_so_one_passing_is_not_luck() -> None:
    for name in reactome_descriptions_info:
        assert sum(p.expect == name for p in PROBES) >= 2, name


def test_a_collection_nobody_chose_fails(capsys) -> None:  # type: ignore[no-untyped-def]
    results = [r for r in _all_collections_covered() if r.probe.expect != "complexes"]
    assert report(results) == 1
    assert "NEVER CHOSEN: complexes" in capsys.readouterr().out


def test_full_coverage_passes(capsys) -> None:  # type: ignore[no-untyped-def]
    assert report(_all_collections_covered()) == 0
    assert "every collection was chosen" in capsys.readouterr().out


def test_leaving_the_selection_open_is_never_a_failure() -> None:
    # Widening is the safe direction and the prompt asks for it when unsure.
    # An open selection must not be reported as a wrong narrow.
    open_one = _result("ewas", [])
    assert open_one.widened
    assert not open_one.wrong


def test_narrowing_to_the_wrong_collection_fails(capsys) -> None:  # type: ignore[no-untyped-def]
    results = _all_collections_covered()
    results.append(_result("disease_variants", ["summations"]))
    assert report(results) == 1
    assert "narrowed away from disease_variants" in capsys.readouterr().out


def test_an_error_is_reported_not_raised(capsys) -> None:  # type: ignore[no-untyped-def]
    results = _all_collections_covered()
    failed = _result("ewas", [])
    failed.error = "RuntimeError: upstream died"
    results.append(failed)
    assert report(results) == 1
    assert "upstream died" in capsys.readouterr().out
