#!/bin/bash
# Move into the app directory
cd "$(dirname "$0")" || exit 1

# Load the .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  No .env file found! Please copy .env.example to .env and add your API key."
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

# Run the Python app
python3 ai_app.py
