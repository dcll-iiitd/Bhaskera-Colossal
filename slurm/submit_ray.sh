#!/bin/bash
# =============================================================================
# Bhaskera — Ray-on-SLURM submission script
# Param Rudra HPC
#
# How this differs from submit_multinode.sh (SLURM backend):
#   - Only ONE process runs per node (not one per GPU)
#   - Node 0 (this script's process) bootstraps the Ray head
#   - The Ray entry point then starts Ray workers on other nodes via srun
#   - TorchTrainer spawns one Ray actor per GPU inside the cluster
#
# Usage:
#   sbatch slurm/submit_ray.sh --config config_ray.yaml --num-workers 4
# =============================================================================

#SBATCH --job-name=bhaskera_ray
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # ONE task per node — Ray manages GPU processes
#SBATCH --gres=gpu:2                 # GPUs per node (Ray will use all of them)
#SBATCH --cpus-per-task=20           # Enough CPUs for Ray workers + head
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/bhaskera_ray_%j_%N.out
#SBATCH --error=logs/bhaskera_ray_%j_%N.err
#SBATCH --exclusive

# =============================================================================
# Parse args passed to sbatch script (forwarded to ray_entry)
# =============================================================================
EXTRA_ARGS="$@"

# =============================================================================
# Environment
# =============================================================================
spack load /lvol4vd
source /scratch/ldls-iiitd/training-framework/venv/bin/activate
export PYTHONPATH="/scratch/ldls-iiitd/training-framework:$PYTHONPATH"

# =============================================================================
# NCCL / networking (same as SLURM path)
# =============================================================================
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

# =============================================================================
# Ray cluster settings
# =============================================================================
# Port for Ray head. Derived from job ID to avoid collisions with other jobs.
export RAY_PORT=$(( 6379 + (SLURM_JOB_ID % 1000) ))

# Optional: set a password for the Redis store
# export RAY_REDIS_PASSWORD="your_password_here"

# Ray dashboard port (for monitoring — accessible via SSH tunnel)
export RAY_DASHBOARD_PORT=$(( 8265 + (SLURM_JOB_ID % 1000) ))

# =============================================================================
# Compute addresses and world size
# =============================================================================
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
WORLD_SIZE=$(( SLURM_NNODES * $(echo $SLURM_GPUS_ON_NODE | cut -d: -f2 2>/dev/null || echo 2) ))

export HEAD_NODE WORLD_SIZE

echo "========================================"
echo "  Bhaskera Ray-on-SLURM Training"
echo "========================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Nodes       : $SLURM_NNODES"
echo "  GPUs/node   : $SLURM_GPUS_ON_NODE"
echo "  World size  : $WORLD_SIZE"
echo "  Head node   : $HEAD_NODE"
echo "  Ray port    : $RAY_PORT"
echo "  Args        : $EXTRA_ARGS"
echo "========================================"

mkdir -p logs ray_results

# =============================================================================
# Launch Ray entry point on the head node only.
#
# ray_entry.py will:
#   1. Start Ray head on this node
#   2. Use `srun --relative=1` to start Ray workers on other nodes
#   3. Call ray.init() to connect to the cluster
#   4. Launch TorchTrainer with --num-workers = WORLD_SIZE
# =============================================================================
python -m bhaskera.launcher.main \
    --launcher ray \
    --num-workers "$WORLD_SIZE" \
    $EXTRA_ARGS