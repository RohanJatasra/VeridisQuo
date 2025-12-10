#!/bin/bash
echo "📦 Installing dependencies..."

cd "$(dirname "$0")/../deepfake-detector" || exit

poetry install
poetry run python main.py