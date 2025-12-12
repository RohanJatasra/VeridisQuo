#!/bin/bash

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install it first."
    exit 1
fi

poetry check --lock || poetry lock #ensures lock file is up to date
poetry install --with dev

poetry run pytest tests/ -v
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed."
fi
exit $EXIT_CODE
