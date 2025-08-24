#!/bin/bash

# Setup script untuk Influencer Recommendation API

echo "🚀 Setting up Influencer Recommendation API..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "🎉 Setup completed!"
    echo ""
    echo "To run the API:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run the application: python app.py"
    echo ""
    echo "API will be available at: http://localhost:5000"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi
