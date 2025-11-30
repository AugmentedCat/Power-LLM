FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/AugmentedCat/Power-LLM.git

# Set working directory to the project
WORKDIR /workspace/Power-LLM

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the GPT2 package
RUN pip install -e GPT2/

# Create necessary directories
RUN mkdir -p GPT2/build GPT2/checkpoints GPT2/outputs

# Set default command
CMD ["/bin/bash"]
