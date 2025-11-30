@echo off
REM Comprehensive analysis of trained model

echo ====================================
echo Trained Model Analysis
echo ====================================
echo.

REM Activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found
    pause
    exit /b 1
)

echo.
echo Running comprehensive model analysis...
echo.
echo This will analyze:
echo   1. Train vs Eval loss over 100 batches each (3,200 sequences)
echo   2. Weight magnitudes per layer
echo   3. Power-ReLU activation patterns
echo   4. Logit distribution and calibration
echo   5. Final performance summary
echo.
echo This may take 5-10 minutes...
echo.

python analyze_trained_model.py

echo.
echo ====================================
echo Analysis Complete
echo ====================================
echo.
pause
