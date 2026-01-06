#!/bin/bash
cd "$(dirname "$0")/.." || exit

echo "📦 Installing dependencies..."
uv sync --extra dev --extra api

echo "🚀 Running the application..."
uv run python src/launch.py