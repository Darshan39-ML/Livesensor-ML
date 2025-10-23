#!/usr/bin/env bash
set -euo pipefail

LOGFILE="/workspaces/Livesensor-ML/.streamlit_run.log"

# Start Streamlit in background
nohup streamlit run /workspaces/Livesensor-ML/streamlit_app.py --server.port 8501 --server.enableCORS false > "$LOGFILE" 2>&1 &
ST_PID=$!

# Wait for server to start
sleep 2

# Poll until server responds or timeout
for i in {1..15}; do
  if curl -sSf http://127.0.0.1:8501/ >/dev/null 2>&1; then
    echo "Streamlit is up"
    break
  fi
  sleep 1
done

# Run Playwright test
python3 /workspaces/Livesensor-ML/tests/smoke_playwright.py
TEST_EXIT=$?

# Stop Streamlit
kill $ST_PID || true

exit $TEST_EXIT
