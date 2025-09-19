#!/bin/sh
set -e
echo "Starting Fingrow Hybrid backend..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
