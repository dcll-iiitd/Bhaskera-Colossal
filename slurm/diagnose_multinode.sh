#!/bin/bash
# =============================================================================
# diagnose_multinode.sh
#
# Lightweight SLURM job that verifies your multi-node setup is correct
# BEFORE you submit a full training run.  Takes ~2 minutes.
#
# Usage:
#   sbatch slurm/diagnose_multinode.sh
# =============================================================================

#SBATCH --job-name=bhaskera_diag
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --output=logs/diag_%j_%N.out
#SBATCH --error=logs/diag_%j_%N.err

spack load /lvol4vd
source /scratch/ldls-iiitd/training-framework/venv/bin/activate
export PYTHONPATH="/scratch/ldls-iiitd/training-framework:$PYTHONPATH"

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 45000) ))
WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export MASTER_ADDR MASTER_PORT WORLD_SIZE

echo "Master: $MASTER_ADDR:$MASTER_PORT  |  World: $WORLD_SIZE"
mkdir -p logs

srun --label python -m bhaskera.launcher.diagnostics