@echo off
REM GPT2 Model Training Script

echo ====================================
echo GPT2 Model Training
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

REM Install GPT2 package if not already installed
echo Installing/Updating GPT2 package and dependencies...
pip install -e . -q
if errorlevel 1 (
    echo ERROR: Failed to install GPT2 package
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "build\corpus.train.txt" (
    echo ERROR: Training corpus not found at build\corpus.train.txt
    echo Please run prepare_openwebtext.bat first to prepare your dataset.
    pause
    exit /b 1
)

if not exist "build\corpus.test.txt" (
    echo ERROR: Evaluation corpus not found at build\corpus.test.txt
    echo Please run prepare_openwebtext.bat first to prepare your dataset.
    pause
    exit /b 1
)

echo.
echo ====================================
echo Training Configuration
echo ====================================
echo.

REM Training parameters - adjust these based on your needs and hardware
set TRAIN_CORPUS=build\corpus.train.txt
set EVAL_CORPUS=build\corpus.test.txt
REM VOCAB_PATH no longer needed - using GPT-2 pretrained vocabulary
set SAVE_MODEL=gpt2-pretrained.pth
set SAVE_CHECKPOINT=ckpt-gpt2.pth

REM Model configuration (adjust for smaller/larger models)
REM IMPORTANT: DIMS must be divisible by HEADS (e.g., 768/12=64, 576/12=48, 384/12=32)
REM For 8GB VRAM: Try DIMS=768, BATCH=32 (GPT2-small size) or DIMS=576, BATCH=48
set SEQ_LEN=1024
set LAYERS=12
set HEADS=12
set DIMS=480
set BATCH_TRAIN=4
set BATCH_EVAL=4
set DROPOUT=0
set USE_RELU=1
set LEARNING_RATE=0.00001
set POST_POWER_NORM=0

REM Training steps
set TOTAL_STEPS=100000
set EVAL_STEPS=500
set SAVE_STEPS=5000

echo Train Corpus: %TRAIN_CORPUS%
echo Eval Corpus: %EVAL_CORPUS%
echo Vocabulary: GPT-2 pretrained (50,257 tokens)
echo.
echo Model Configuration:
echo - Sequence Length: %SEQ_LEN%
echo - Layers: %LAYERS%
echo - Attention Heads: %HEADS%
echo - Dimensions: %DIMS%
echo - Training Batch Size: %BATCH_TRAIN%
echo - Eval Batch Size: %BATCH_EVAL%
echo - Dropout: %DROPOUT%
echo - Use ReLU: %USE_RELU% (1=enabled, 0=disabled - pure power scaling)
echo - Learning Rate: %LEARNING_RATE%
echo - Post-Power Norm: %POST_POWER_NORM% (1=enabled, 0=disabled - prevents activation explosion)
echo.
echo Training Steps:
echo - Total Steps: %TOTAL_STEPS%
echo - Eval Every: %EVAL_STEPS% steps
echo - Save Every: %SAVE_STEPS% steps
echo.
echo Output Files:
echo - Model: %SAVE_MODEL%
echo - Checkpoint: %SAVE_CHECKPOINT%
echo.
echo ====================================
echo.

echo Starting training... (Press Ctrl+C to stop)
echo.

REM Start training
REM Build base command
set BASE_CMD=python -m gpt2 train --train_corpus %TRAIN_CORPUS% --eval_corpus %EVAL_CORPUS% --save_model_path %SAVE_MODEL% --save_checkpoint_path %SAVE_CHECKPOINT% --seq_len %SEQ_LEN% --layers %LAYERS% --heads %HEADS% --dims %DIMS% --batch_train %BATCH_TRAIN% --batch_eval %BATCH_EVAL% --dropout %DROPOUT% --base_lr %LEARNING_RATE% --total_steps %TOTAL_STEPS% --eval_steps %EVAL_STEPS% --save_steps %SAVE_STEPS%

REM Add optional flags
set OPTIONAL_FLAGS=
if "%USE_RELU%"=="0" set OPTIONAL_FLAGS=%OPTIONAL_FLAGS% --no_relu
if "%POST_POWER_NORM%"=="1" set OPTIONAL_FLAGS=%OPTIONAL_FLAGS% --use_post_power_norm

REM Execute command
%BASE_CMD%%OPTIONAL_FLAGS%

echo.
echo ====================================
echo Training Complete!
echo ====================================
echo.
echo Model saved to: %SAVE_MODEL%
echo Checkpoint saved to: %SAVE_CHECKPOINT%
echo.
echo To resume training from checkpoint, add this option:
echo --from_checkpoint %SAVE_CHECKPOINT%
echo.
echo To generate sentences with your trained model:
echo Run run_generate.bat (update MODEL_PATH to %SAVE_MODEL%)
echo.

pause
