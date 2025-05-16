@echo off
echo Installing required packages...
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install packages with pip. Trying with trusted hosts...
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo.
        echo ERROR: Failed to install packages. Please install Python and make sure it's in your PATH.
        echo Visit https://www.python.org/downloads/ to download Python.
        echo Be sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit application...
echo.
python -m streamlit run churn_prediction_app.py -- --model_path=RandomForest_best_model.pkl
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to start Streamlit application.
    echo.
    echo If Python is installed but not in PATH, try:
    echo python -m streamlit run churn_prediction_app.py -- --model_path=RandomForest_best_model.pkl
    echo.
    pause
)
