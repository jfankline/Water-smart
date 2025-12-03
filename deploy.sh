#!/bin/bash

# Deployment script for WaterSmart Backend

echo "🚀 Starting WaterSmart Backend Deployment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please update .env file with your API keys"
fi

# Initialize database
echo "Initializing database..."
python -c "
from app import app, init_db
with app.app_context():
    init_db()
    print('Database initialized successfully')
"

# Run tests
echo "Running tests..."
python -m pytest test_api.py -v

# Start server
echo "Starting Flask server..."
echo "🌐 Server running on http://localhost:5000"
echo "📚 API Documentation available at http://localhost:5000/"

if [ "$1" = "production" ]; then
    echo "Starting production server with gunicorn..."
    gunicorn --bind 0.0.0.0:5000 app:app
else
    echo "Starting development server..."
    python app.py
fi