@echo off
REM Download and prepare C4 dataset for GPT2 training

echo ====================================
echo C4 Dataset Preparation
echo ====================================
echo.

REM Activate virtual environment
if exist "..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ..\venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at ..\venv\Scripts\activate.bat
    echo Please create a virtual environment first or update the path.
    pause
    exit /b 1
)

echo.
echo This script will download and prepare a large text dataset.
echo.
echo Dataset: C4 (Colossal Clean Crawled Corpus)
echo WARNING: This will download approximately 20-25GB of data.
echo Processing may take 30-90 minutes depending on your system.
echo.
echo The dataset will be tokenized using GPT-2 BPE tokenizer.
echo C4 is more diverse than Wikipedia - better for benchmarks!
echo.
set /p CONFIRM="Continue with download? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.
echo Downloading and preparing C4 dataset...
echo Using 34 CPU workers for parallel processing...
echo This will take a while. Please be patient...
echo.

REM Suppress tokenizer warnings by setting environment variable
set TRANSFORMERS_VERBOSITY=error
python download_c4_limited.py --output-dir build --seq-len 128 --num-workers 34

if errorlevel 1 (
    echo.
    echo ERROR: Failed to prepare dataset.
    echo Make sure you have:
    echo   - Internet connection
    echo   - Sufficient disk space (~50GB free recommended)
    echo   - transformers and datasets packages installed
    pause
    exit /b 1
)

echo.
echo ====================================
echo SUCCESS!
echo ====================================
echo.
echo C4 dataset prepared in 'build\' directory.
echo.
echo Files created:
echo   - corpus.train.txt  (training data)
echo   - corpus.test.txt   (test data)
echo.
echo Next step: Run train_model.bat to start training!
echo.

pause
