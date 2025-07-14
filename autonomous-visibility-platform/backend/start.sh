#!/bin/bash

echo "🚀 Starting Autonomous Visibility Platform API..."

# Find Python command
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Found python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Found python"
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Using Python: $PYTHON_CMD"

# Create necessary directories
mkdir -p data/duckdb
mkdir -p logs

# Install minimal dependencies
echo "📦 Installing basic dependencies..."
$PYTHON_CMD -m pip install --user fastapi uvicorn pydantic

# Start the API server
echo "🌐 Starting FastAPI server on port 8000..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

cd app
$PYTHON_CMD -c "
import uvicorn
uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
"
