import pytest

from util.config_yml.user_matching import match_user

LOGGED_IN = "someone@example.org"
GUEST = None


@pytest.mark.parametrize("spec", [None, []])
def test_empty_spec_matches_everyone(spec: list[str] | None) -> None:
    assert match_user(spec, LOGGED_IN) is True
    assert match_user(spec, GUEST) is True


def test_all_matches_everyone() -> None:
    assert match_user(["all"], LOGGED_IN) is True
    assert match_user(["all"], GUEST) is True


def test_guests_and_logged_in_are_mutually_exclusive() -> None:
    assert match_user(["guests"], GUEST) is True
    assert match_user(["guests"], LOGGED_IN) is False
    assert match_user(["logged_in"], LOGGED_IN) is True
    assert match_user(["logged_in"], GUEST) is False


def test_glob_patterns_match_on_the_full_identifier() -> None:
    assert match_user(["*@gmail.com"], "person@gmail.com") is True
    assert match_user(["*@gmail.com"], "person@oicr.on.ca") is False
    assert match_user(["*@oicr.on.ca", "*@gmail.com"], "person@gmail.com") is True


def test_slash_delimited_entries_are_treated_as_regex() -> None:
    """Undocumented: /.../ entries are regex, which `.config.schema.yaml` disallows."""
    assert match_user(["/^admin-/"], "admin-jane") is True
    assert match_user(["/^admin-/"], "jane-admin") is False
    # note: re.search, not fullmatch -- the pattern is unanchored by default
    assert match_user(["/oicr/"], "person@oicr.on.ca") is True


def test_guest_never_matches_identifier_patterns() -> None:
    assert match_user(["*@gmail.com"], GUEST) is False
    assert match_user(["logged_in", "*@gmail.com"], GUEST) is False


def test_empty_entry_is_skipped_not_a_crash() -> None:
    """`entry[0]` used to index without a length check, so `users: [""]` raised."""
    assert match_user([""], LOGGED_IN) is False
    assert match_user([""], GUEST) is False
    assert (
        match_user(["", "all"], LOGGED_IN) is True
    ), "a real entry after it still counts"
