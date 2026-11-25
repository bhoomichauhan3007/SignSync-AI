#!/usr/bin/env bash
# Runs every test suite that can run in the current environment.
set -u
cd "$(dirname "$0")/../backend"
status=0
echo "== RAG tutor (Member 4)";            python -m unittest discover -s rag/tests -t . || status=1
echo "== Cross-member integration (2, 3, backend adapter)"; python -m unittest discover -s integration_tests -t . || status=1
if python -c "import pytest, fastapi" 2>/dev/null; then
  echo "== Backend API suite (Member 5)"; python -m pytest tests -q || status=1
else
  echo "== Backend API suite (Member 5): SKIPPED (needs fastapi + pytest; run: pip install -r ../requirements.txt)"
fi
exit $status
