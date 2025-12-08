#!/bin/bash
#SBATCH --job-name=vllm-es        # Job name for identification
#SBATCH --output=vllm-es_%j.out   # Standard output log file (%j = Job ID)
#SBATCH --error=vllm-es_%j.err    # Standard error log file (%j = Job ID)
#SBATCH --partition=compsci-gpu   # Partition (copied from your job output)
#SBATCH --nodes=1                 # Run all processes on a single node
#SBATCH --gres=gpu:a5000:4        # Request 2 A6000 GPUs
#SBATCH --cpus-per-gpu=6          # Request 6 CPUs per GPU (12 total)
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-01:00:00         # Time limit (e.g., 4 days)

echo "---"
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "---"

# --- 1. Activate Virtual Environment (venv) ---
echo "Activating Python venv from .venv/"
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    echo "Make sure '.venv/bin/activate' exists relative to your script."
    exit 1
fi

echo "Virtual environment activated."

# --- 2. Set Environment Variables for your Python script ---
# This lets your Python script know how many resources Slurm gave it.
export NUM_GPUS_FROM_SLURM=$SLURM_GPUS_ON_NODE
export CPUS_PER_GPU_FROM_SLURM=$SLURM_CPUS_PER_GPU

# --- 3. Run the Python Script ---
# IMPORTANT: Change 'your_script_name.py' to your actual file name
echo "Running Python script..."
python train.py

echo "---"
echo "Job finished."
echo "---"