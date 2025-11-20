#!/usr/bin/env bash
# Small helper to run the app locally with env from .env (if present)
set -euo pipefail
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi
streamlit run streamlit_app.py
