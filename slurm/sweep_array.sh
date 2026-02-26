#!/bin/bash
# =============================================================================
# Bhaskera SLURM Array Job — Hyperparameter Sweep
#
# Runs multiple configs in parallel across SLURM array tasks.
# Each array task reads its config from the CONFIGS array below.
#
# Usage:
#   sbatch sweep_array.sh
#   sbatch --array=0-1 sweep_array.sh   # only tasks 0 and 1
# =============================================================================

#SBATCH --job-name=bhaskera_sweep
#SBATCH --array=0-2                    # indices into CONFIGS array below
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_%A_%a_%N.out   # %A=array_job_id, %a=task_id
#SBATCH --error=logs/sweep_%A_%a_%N.err
#SBATCH --exclusive

# ---- configs to sweep -------------------------------------------------------
CONFIGS=(
    "config_multinode.yaml"
    "config_multinode_lr1e4.yaml"
    "config_multinode_r32.yaml"
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task $SLURM_ARRAY_TASK_ID -> config: $CONFIG"

# ---- environment ------------------------------------------------------------
spack load /lvol4vd
source /scratch/ldls-iiitd/training-framework/venv/bin/activate
export PYTHONPATH="/scratch/ldls-iiitd/training-framework:$PYTHONPATH"

# ---- networking -------------------------------------------------------------
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 45000) + SLURM_ARRAY_TASK_ID ))
export MASTER_ADDR MASTER_PORT
WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export WORLD_SIZE

mkdir -p logs

srun --label --kill-on-bad-exit=1 \
    python -m bhaskera.launcher.slurm_entry --config "$CONFIG"