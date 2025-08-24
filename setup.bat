@echo off
REM Setup script untuk Influencer Recommendation API - Windows

echo 🚀 Setting up Influencer Recommendation API...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Show Python version
echo ✅ Python version:
python --version

REM Create virtual environment if not exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing requirements...
pip install -r requirements.txt

REM Check if installation was successful
if %errorlevel% == 0 (
    echo ✅ All dependencies installed successfully!
    echo.
    echo 🎉 Setup completed!
    echo.
    echo To run the API:
    echo 1. Activate virtual environment: venv\Scripts\activate
    echo 2. Run the application: python app.py
    echo.
    echo API will be available at: http://localhost:5000
) else (
    echo ❌ Failed to install dependencies. Please check the error messages above.
    pause
    exit /b 1
)

pause
