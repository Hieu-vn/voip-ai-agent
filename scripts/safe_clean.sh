#!/usr/bin/env bash
set -euo pipefail

PROJECT="voip-ai-agent"
APPLY=0
DEEP=0   # deep=1 sẽ xoá cả anonymous volumes & cache lớn
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --deep)  DEEP=1; shift ;;
    --project) PROJECT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "==> Project: $PROJECT | apply=$APPLY | deep=$DEEP"
echo "==> Disk usage before:"
docker system df || true
echo

echo "==> Stop stack (no volumes):"
set -x
docker compose --project-name "$PROJECT" down --remove-orphans || true
set +x
echo

echo "==> List project containers/images/volumes (preview):"
docker ps -a --filter "label=com.docker.compose.project=${PROJECT}" || true
docker images --filter "label=com.docker.compose.project=${PROJECT}" || true
docker volume ls --filter "label=com.docker.compose.project=${PROJECT}" || true
echo

echo "==> Prune build cache & dangling images (safe):"
if [[ $APPLY -eq 1 ]]; then
  docker image prune -f
  docker builder prune -f
  docker network prune -f
else
  echo "  [dry-run] docker image prune -f"
  echo "  [dry-run] docker builder prune -f"
  echo "  [dry-run] docker network prune -f"
fi
echo

if [[ $DEEP -eq 1 ]]; then
  echo "==> Deep clean (anonymous volumes & buildx cache):"
  if [[ $APPLY -eq 1 ]]; then
    # Chỉ xoá volumes vô chủ (dangling). Named volumes KHÔNG bị xoá nếu đang gắn.
    docker volume prune -f
    docker buildx prune -f
  else
    echo "  [dry-run] docker volume prune -f"
    echo "  [dry-run] docker buildx prune -f"
  fi
fi

echo
echo "==> Optional: clean workspace caches (Python/__pycache__) [non-destructive to code]"
if [[ $APPLY -eq 1 ]]; then
  find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
  find . -type f -name "*.pyc" -delete 2>/dev/null || true
else
  echo "  [dry-run] find . -type d -name '__pycache__' -prune -exec rm -rf {} +"
  echo "  [dry-run] find . -type f -name '*.pyc' -delete"
fi

echo
echo "==> Disk usage after (estimate):"
docker system df || true
echo "Done."
