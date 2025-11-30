# Running GPT-2 Training on Vast.ai

This guide will help you set up and run your GPT-2 training on Vast.ai cloud GPU instances.

## Prerequisites

1. Create a Vast.ai account at https://vast.ai
2. Add credits to your account
3. Have this GitHub repository ready: https://github.com/AugmentedCat/Power-LLM.git

## Step 1: Rent a GPU Instance

1. Go to https://vast.ai/console/create/
2. Search for instances with:
   - **GPU**: NVIDIA RTX 3090, RTX 4090, or A6000 (recommended for 24GB VRAM)
   - **Disk Space**: At least 50GB
   - **Image**: Select "PyTorch" template (pytorch/pytorch)
3. Click "Rent" on your chosen instance

## Step 2: Connect to Your Instance

Once your instance is running:

```bash
# Use the SSH command provided by Vast.ai (looks like):
ssh -p <PORT> root@<IP_ADDRESS>
```

## Step 3: Setup Your Environment

Run these commands on your Vast.ai instance:

```bash
# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/AugmentedCat/Power-LLM.git
cd Power-LLM

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the GPT2 package
cd GPT2
pip install -e .
```

## Step 4: Download Dataset

Choose one of these datasets:

### Option A: WikiText (smaller, faster)
```bash
python download_wikitext.py
```

### Option B: C4 Dataset (limited)
```bash
python download_c4_limited.py
```

This will create:
- `build/corpus.train.txt` - Training data
- `build/corpus.test.txt` - Evaluation data

## Step 5: Configure Training Parameters

Edit `train.sh` to adjust parameters based on your GPU:

```bash
nano train.sh
```

Recommended settings by GPU memory:
- **8GB VRAM**: `DIMS=768`, `BATCH_TRAIN=32`
- **16GB VRAM**: `DIMS=1024`, `BATCH_TRAIN=64`
- **24GB+ VRAM**: `DIMS=1280`, `BATCH_TRAIN=96`

## Step 6: Start Training

```bash
# Make the script executable
chmod +x train.sh

# Start training
bash train.sh
```

Training will run and save:
- **Model**: `gpt2-pretrained.pth` - Final trained model
- **Checkpoint**: `ckpt-gpt2.pth` - Periodic checkpoints for resuming

## Monitoring Training

The training script will show:
- Current step
- Training loss
- Evaluation perplexity
- Time per step

Press `Ctrl+C` to stop training at any time.

## Resume Training from Checkpoint

If training stops, resume with:

```bash
# Edit train.sh and add to the CMD line:
--from_checkpoint ckpt-gpt2.pth
```

## Download Your Trained Model

From your local machine:

```bash
# Use the SCP command (replace with your Vast.ai connection info)
scp -P <PORT> root@<IP_ADDRESS>:/workspace/Power-LLM/GPT2/gpt2-pretrained.pth ./
```

## Troubleshooting

### Out of Memory Error
- Reduce `BATCH_TRAIN` and `BATCH_EVAL` in `train.sh`
- Reduce `DIMS` (must be divisible by `HEADS`)
- Reduce `SEQ_LEN`

### Dataset Not Found
- Make sure you ran the download script
- Check that files exist: `ls -la build/`

### CUDA Not Available
- Verify GPU instance has CUDA: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## Cost Optimization

- **Checkpoints**: Save frequently (every 5000 steps) so you can stop/resume
- **Monitoring**: Check training progress regularly to avoid running unnecessarily
- **Auto-stop**: Consider setting up scripts to auto-stop after N steps
- **Spot instances**: Use interruptible instances for lower cost (save checkpoints frequently!)

## Instance Management

Remember to **destroy** your instance when done to stop billing:
1. Go to Vast.ai console
2. Find your instance
3. Click "Destroy"

Happy training!
