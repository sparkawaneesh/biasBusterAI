#!/bin/bash

echo "🔄 Stopping existing servers..."
pkill -f "tsx server/index.ts" 2>/dev/null || true
pkill -f "node" 2>/dev/null || true
sleep 2

echo "🚀 Starting Python Backend..."
cd python_backend

# Export GROQ API key
export GROQ_API_KEY="$GROQ_API_KEY"

# Start Python Flask server
python3 run.py