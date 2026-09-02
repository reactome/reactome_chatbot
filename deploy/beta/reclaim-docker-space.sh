#!/usr/bin/env bash
#
# Frees disk on dev.reactome.org so the chatbot image + embeddings bundle fit.
#
#   sudo ~/reclaim-docker-space.sh            # audit only, changes NOTHING
#   sudo ~/reclaim-docker-space.sh --safe     # dangling images + build cache
#   sudo ~/reclaim-docker-space.sh --all      # the above, plus dangling volumes
#
# This host serves both dev.reactome.org and the origin behind beta.reactome.org
# on one 88G volume. Filling it takes both down, which is why the default mode
# changes nothing and why --all is a separate, deliberate flag.
#
# --safe touches only things Docker can rebuild: layers no image references any
# more, and build cache. --all additionally removes ANONYMOUS volumes that no
# container references. Read the audit before using it; a removed volume is gone.

set -euo pipefail

MODE="${1:-audit}"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }

free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }

BEFORE=$(free_gb)
say "Disk before: ${BEFORE}G free"
df -h / | tail -1

say "What is reclaimable"
docker system df

say "Dangling volumes (candidates for --all)"
found=0
DANGLING=()
for v in $(docker volume ls -qf dangling=true); do
  found=1
  DANGLING+=("$v")
  mp=$(docker volume inspect -f '{{.Mountpoint}}' "$v")
  created=$(docker volume inspect -f '{{.CreatedAt}}' "$v")
  size=$(du -sh "$mp" 2>/dev/null | cut -f1 || echo '?')
  printf '\n  %s\n    created %s, %s\n' "$v" "$created" "$size"
  printf '    top-level contents:\n'
  ls -A "$mp" 2>/dev/null | head -8 | sed 's/^/      /' || true
  n=$(ls -A "$mp" 2>/dev/null | wc -l)
  [ "$n" -gt 8 ] && printf '      ... and %s more entries\n' "$((n - 8))"
done
[ "$found" -eq 0 ] && ok "none"

if [ "$MODE" = "audit" ]; then
  say "Audit only -- nothing was changed."
  echo "  Re-run with --safe (images + build cache) or --all (also the volumes above)."
  exit 0
fi

say "Reclaiming: dangling images"
docker image prune -f
say "Reclaiming: build cache"
docker builder prune -f

if [ "$MODE" = "--all" ]; then
  say "Reclaiming: dangling volumes"
  if [ ${#DANGLING[@]} -gt 0 ]; then
    # Remove by id rather than `docker volume prune`, which only sweeps volumes
    # Docker tagged as anonymous and silently leaves older unreferenced ones.
    docker volume rm "${DANGLING[@]}" || warn "some volumes could not be removed"
  else
    ok "no dangling volumes"
  fi
elif [ "$MODE" != "--safe" ]; then
  warn "unknown mode '$MODE' -- treated as --safe"
fi

AFTER=$(free_gb)
say "Disk after: ${AFTER}G free (was ${BEFORE}G)"
df -h / | tail -1

# Peak requirement: ~3.2G image + 1.3G zip + ~3.0G extracted bundle.
if [ "$AFTER" -lt 9 ]; then
  warn "under 9G free; the pull + bundle extract needs roughly 7.5G at peak."
  if [ "$MODE" = "--all" ]; then
    warn "already ran --all; the remaining space has to come from outside Docker."
  else
    warn "try --all, or free space outside Docker."
  fi
else
  ok "enough headroom for the image and the Release95 bundle"
fi
