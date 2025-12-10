#!/bin/bash

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install it first."
    exit 1
fi

poetry install --with dev

poetry run pytest tests/ -v

echo "✅ All tests passed!"
