#!/usr/bin/env bash
# run_fl.sh - robust launcher for the federated demo (sanity checks, data prep, server & clients)
set -euo pipefail

# ---------- helpers ----------
log()   { printf "\n[ %s ] %s\n" "$(date +'%H:%M:%S')" "$*"; }
err()   { printf "\n[ %s ] ERROR: %s\n" "$(date +'%H:%M:%S')" "$*" >&2; }

# declare CLIENT_PIDS early so cleanup never sees it undefined
CLIENT_PIDS=()
SERVER_PID=""

cleanup() {
  log "Cleaning up processes..."
  # kill clients if any
  if [ "${#CLIENT_PIDS[@]}" -gt 0 ]; then
    for p in "${CLIENT_PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then
        log "Killing client pid $p"
        kill "$p" || true
      fi
    done
  fi
  # kill server if running
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Killing server pid $SERVER_PID"
    kill "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

# ---------- find repo root ----------
REPO_ROOT="$(pwd)"
# if inside git repo, prefer git toplevel
if command -v git >/dev/null 2>&1; then
  GIT_TOPLEVEL=$(git rev-parse --show-toplevel 2>/dev/null || true)
  if [ -n "$GIT_TOPLEVEL" ]; then
    REPO_ROOT="$GIT_TOPLEVEL"
  fi
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
log "Working directory set to: $REPO_ROOT"
log "PYTHONPATH = $PYTHONPATH"

# ---------- sanity check ----------
SANITY_SCRIPT="$REPO_ROOT/scripts/sanity_check.py"
if [ ! -f "$SANITY_SCRIPT" ]; then
  # fallback search
  SANITY_SCRIPT=$(find "$REPO_ROOT" -maxdepth 3 -type f -name "sanity_check.py" -print -quit || true)
fi
if [ -z "$SANITY_SCRIPT" ] || [ ! -f "$SANITY_SCRIPT" ]; then
  err "Sanity check script not found. Expected at scripts/sanity_check.py (or nearby)."
  exit 1
fi

log "Running sanity check: $SANITY_SCRIPT"
if python "$SANITY_SCRIPT"; then
  log "Sanity check passed."
else
  err "Sanity check failed. Fix issues and re-run."
  exit 1
fi

# ---------- checks ----------
REQUIRED_DIRS=( "server" "clients" "data" )
for d in "${REQUIRED_DIRS[@]}"; do
  if [ ! -d "$REPO_ROOT/$d" ]; then
    err "Required directory missing: $REPO_ROOT/$d"
    exit 1
  fi
done

if [ ! -f "$REPO_ROOT/data/download.py" ]; then
  err "Missing: data/download.py"
  exit 1
fi

# ---------- data download (skip if present) ----------
DATA_PRESENT=false
if [ -d "$REPO_ROOT/clients/node1/data/images" ] && [ "$(ls -A "$REPO_ROOT/clients/node1/data/images" 2>/dev/null | wc -l)" -gt 0 ]; then
  DATA_PRESENT=true
fi

if [ "$DATA_PRESENT" = true ]; then
  log "Data already present; skipping download."
else
  log "Downloading/extracting node datasets..."
  python "$REPO_ROOT/data/download.py"
  log "Data download/extract completed."
fi

# ---------- start server ----------
log "Starting Flower server..."
python "$REPO_ROOT/server/server_flower.py" &
SERVER_PID=$!
log "Server started with PID $SERVER_PID"

# ---------- wait for server to listen ----------
SERVER_ADDR="127.0.0.1"
SERVER_PORT=8080
MAX_WAIT_SEC=30
SLEEP_INTERVAL=0.5
elapsed=0

check_port() {
  python - <<PY - "$SERVER_ADDR" "$SERVER_PORT"
import socket, sys
addr=sys.argv[1]; port=int(sys.argv[2])
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect((addr, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

log "Waiting for server to listen on ${SERVER_ADDR}:${SERVER_PORT} (timeout ${MAX_WAIT_SEC}s)..."
while ! check_port; do
  sleep "$SLEEP_INTERVAL"
  elapsed=$(awk "BEGIN {print $elapsed+$SLEEP_INTERVAL}")
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    err "Server process $SERVER_PID died. See server logs."
    exit 1
  fi
  if (( $(echo "$elapsed >= $MAX_WAIT_SEC" | bc -l) )); then
    err "Timeout waiting for server port. Exiting."
    exit 1
  fi
done
log "Server is listening on ${SERVER_ADDR}:${SERVER_PORT}"

# ---------- start clients ----------
: "${NODES_TO_RUN:=1,2,3}"
IFS=',' read -ra NODE_IDS <<< "$NODES_TO_RUN"
log "Nodes to run: ${NODE_IDS[*]}"

for id in "${NODE_IDS[@]}"; do
  CLIENT_SCRIPT="$REPO_ROOT/clients/node${id}/client_flower.py"
  if [ ! -f "$CLIENT_SCRIPT" ]; then
    err "Missing client script: $CLIENT_SCRIPT"
    kill "$SERVER_PID" || true
    exit 1
  fi
  log "Starting client node$id"
  python "$CLIENT_SCRIPT" &
  pid=$!
  CLIENT_PIDS+=("$pid")
  log "Client node$id started with pid $pid"
  sleep 0.5
done

# ---------- wait for clients ----------
log "Waiting for clients to finish: ${CLIENT_PIDS[*]}"
for p in "${CLIENT_PIDS[@]}"; do
  if kill -0 "$p" 2>/dev/null; then
    wait "$p" || true
    log "Client pid $p finished"
  else
    log "Client pid $p already exited"
  fi
done

# ---------- shutdown server ----------
if kill -0 "$SERVER_PID" 2>/dev/null; then
  log "Terminating server pid $SERVER_PID"
  kill "$SERVER_PID" || true
  sleep 1
else
  log "Server already exited."
fi

log "Federated run completed."
exit 0
