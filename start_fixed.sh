#!/bin/bash

echo "🚀 Starting Autonomous Visibility Platform API..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastapi uvicorn pydantic

# Create necessary directories
mkdir -p data/duckdb logs

# Start the server
echo "🌐 Starting FastAPI server on port 8000..."
echo "✅ Backend will be available at: http://localhost:8000"
echo "📚 API docs will be available at: http://localhost:8000/api/docs"

cd app
python main.py
