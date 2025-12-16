#!/bin/bash

if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first."
    exit 1
fi

cd "$(dirname "$0")/.." || exit

uv sync --extra dev

uv run pytest tests/ -v
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed."
fi
exit $EXIT_CODE
