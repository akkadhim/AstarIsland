#!/bin/bash
# Quick-start script for Astar Island competition
# Usage: ./run.sh YOUR_JWT_TOKEN [--dry-run]

TOKEN=$1
if [ -z "$TOKEN" ]; then
  echo "Usage: ./run.sh YOUR_JWT_TOKEN [--dry-run]"
  echo ""
  echo "Get your token:"
  echo "  1. Log in at app.ainm.no"
  echo "  2. Open browser DevTools → Application → Cookies"
  echo "  3. Copy the 'access_token' value"
  exit 1
fi

DRYRUN=""
if [ "$2" == "--dry-run" ]; then
  DRYRUN="--dry-run"
  echo "[DRY RUN MODE - will not submit]"
fi

echo "Starting Astar Island solver..."
echo "Using 3 GPUs, 30K sims per seed"
echo ""

python3 main.py \
  --token "$TOKEN" \
  --total-sims 30000 \
  --n-eval-sims 300 \
  $DRYRUN

echo ""
echo "Done! Check app.ainm.no for your score."
