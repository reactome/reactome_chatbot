#!/usr/bin/env bash
# Run exactly what CI runs, and say what each step did.
#
# Two failures made this worth having. A commit went out with six mypy
# errors because the command was piped and the pipeline's exit code was the
# pipe's, not mypy's. And PR #290 failed CI lint while `mypy src` passed
# locally -- CI runs bare `mypy`, which covers 159 files rather than 152,
# and the extra seven include the tests.
#
# So: no pipes around the commands, every exit code printed, one verdict at
# the end. Mirrors .github/workflows/ci.yml; if that changes, change this.
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)

declare -a names=() codes=()
step() {
    local name="$1"; shift
    "$@" >/tmp/checks-$$.log 2>&1
    local code=$?
    names+=("$name"); codes+=("$code")
    if [ "$code" -eq 0 ]; then
        printf '  %-16s ok\n' "$name"
    else
        printf '  %-16s FAILED (exit %d)\n' "$name" "$code"
        sed 's/^/      /' /tmp/checks-$$.log | tail -25
    fi
}

echo "checks:"
step "ruff check"  "$PY" -m ruff check .
step "ruff format" "$PY" -m ruff format --check .
step "mypy"        "$PY" -m mypy
step "pytest"      "$PY" -m pytest
rm -f /tmp/checks-$$.log

failed=0
for i in "${!codes[@]}"; do
    [ "${codes[$i]}" -ne 0 ] && failed=$((failed + 1))
done

if [ "$failed" -eq 0 ]; then
    echo "all ${#names[@]} checks passed"
    exit 0
fi
echo "$failed of ${#names[@]} checks FAILED"
exit 1
