#!/bin/bash
cd "$(dirname "$0")/../src" || exit

echo "📦 Installing dependencies..."
poetry install

echo "🚀 Running the application..."
poetry run python main.py