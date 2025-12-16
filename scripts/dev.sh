#!/bin/bash
cd "$(dirname "$0")/.." || exit

echo "📦 Installing dependencies..."
uv sync

echo "🚀 Running the application..."
uv run python src/main.py