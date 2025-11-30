#!/bin/bash
# GPT2 Model Training Script for Linux/Vast.ai

echo "===================================="
echo "GPT2 Model Training"
echo "===================================="
echo ""

# Check if required files exist
if [ ! -f "build/corpus.train.txt" ]; then
    echo "ERROR: Training corpus not found at build/corpus.train.txt"
    echo "Please download and prepare your dataset first."
    echo ""
    echo "Options:"
    echo "  - Run: python download_c4_limited.py"
    echo "  - Run: python download_wikitext.py"
    exit 1
fi

if [ ! -f "build/corpus.test.txt" ]; then
    echo "ERROR: Evaluation corpus not found at build/corpus.test.txt"
    echo "Please download and prepare your dataset first."
    exit 1
fi

echo "===================================="
echo "Training Configuration"
echo "===================================="
echo ""

# Training parameters - adjust these based on your needs and hardware
TRAIN_CORPUS="build/corpus.train.txt"
EVAL_CORPUS="build/corpus.test.txt"
SAVE_MODEL="gpt2-pretrained.pth"
SAVE_CHECKPOINT="ckpt-gpt2.pth"

# Model configuration
# IMPORTANT: DIMS must be divisible by HEADS
# For different GPU memory:
# - 8GB VRAM: DIMS=768, BATCH=32 (GPT2-small)
# - 16GB VRAM: DIMS=1024, BATCH=64
# - 24GB+ VRAM: DIMS=1280, BATCH=96
SEQ_LEN=1024
LAYERS=12
HEADS=12
DIMS=768
BATCH_TRAIN=64
BATCH_EVAL=64
DROPOUT=0
USE_RELU=1
LEARNING_RATE=0.00001
POST_POWER_NORM=0
USE_AMP=1  # Mixed precision (BF16) - 3x faster, 43% less memory

# Training steps
TOTAL_STEPS=1000000
EVAL_STEPS=500
SAVE_STEPS=5000

echo "Train Corpus: $TRAIN_CORPUS"
echo "Eval Corpus: $EVAL_CORPUS"
echo "Vocabulary: GPT-2 pretrained (50,257 tokens)"
echo ""
echo "Model Configuration:"
echo "  - Sequence Length: $SEQ_LEN"
echo "  - Layers: $LAYERS"
echo "  - Attention Heads: $HEADS"
echo "  - Dimensions: $DIMS"
echo "  - Training Batch Size: $BATCH_TRAIN"
echo "  - Eval Batch Size: $BATCH_EVAL"
echo "  - Dropout: $DROPOUT"
echo "  - Use ReLU: $USE_RELU (1=enabled, 0=disabled)"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Post-Power Norm: $POST_POWER_NORM (1=enabled, 0=disabled)"
echo "  - Mixed Precision: $USE_AMP (1=BF16 enabled, 0=FP32)"
echo ""
echo "Training Steps:"
echo "  - Total Steps: $TOTAL_STEPS"
echo "  - Eval Every: $EVAL_STEPS steps"
echo "  - Save Every: $SAVE_STEPS steps"
echo ""
echo "Output Files:"
echo "  - Model: $SAVE_MODEL"
echo "  - Checkpoint: $SAVE_CHECKPOINT"
echo ""
echo "===================================="
echo ""

echo "Starting training... (Press Ctrl+C to stop)"
echo ""

# Build base command
CMD="python -m gpt2 train \
    --train_corpus $TRAIN_CORPUS \
    --eval_corpus $EVAL_CORPUS \
    --save_model_path $SAVE_MODEL \
    --save_checkpoint_path $SAVE_CHECKPOINT \
    --seq_len $SEQ_LEN \
    --layers $LAYERS \
    --heads $HEADS \
    --dims $DIMS \
    --batch_train $BATCH_TRAIN \
    --batch_eval $BATCH_EVAL \
    --dropout $DROPOUT \
    --base_lr $LEARNING_RATE \
    --total_steps $TOTAL_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS"

# Add optional flags
if [ "$USE_RELU" = "0" ]; then
    CMD="$CMD --no_relu"
fi

if [ "$POST_POWER_NORM" = "1" ]; then
    CMD="$CMD --use_post_power_norm"
fi

if [ "$USE_AMP" = "1" ]; then
    CMD="$CMD --use_amp"
fi

# Execute command
eval $CMD

echo ""
echo "===================================="
echo "Training Complete!"
echo "===================================="
echo ""
echo "Model saved to: $SAVE_MODEL"
echo "Checkpoint saved to: $SAVE_CHECKPOINT"
echo ""
echo "To resume training from checkpoint, add:"
echo "  --from_checkpoint $SAVE_CHECKPOINT"
echo ""
