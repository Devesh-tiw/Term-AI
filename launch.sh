#!/bin/bash
cd "$(dirname "$0")" || exit 1

if [ ! -f .env ]; then
    echo "⚠️  No .env file found! Please copy .env.example to .env and add your API key."
    exit 1
fi

source venv/bin/activate
python3 ai_app.py
