#!/bin/bash
# =============================================================================
# Bhaskera Multi-Node Training — SLURM Submission Script
# Param Rudra HPC
#
# Usage:
#   sbatch submit_multinode.sh --config config_fsdp.yaml
#   sbatch --nodes=4 submit_multinode.sh --config config_fsdp.yaml
#
# Override any #SBATCH directive at submission time, e.g.:
#   sbatch --nodes=4 --ntasks-per-node=4 submit_multinode.sh --config my.yaml
# =============================================================================

#SBATCH --job-name=bhaskera_train
#SBATCH --nodes=2                   # Number of nodes (override at sbatch time)
#SBATCH --ntasks-per-node=2         # Must match GPUs per node you request
#SBATCH --gres=gpu:2                # GPUs per node
#SBATCH --cpus-per-task=10          # CPU cores per GPU task
#SBATCH --partition=gpu             # Partition name on Rudra
#SBATCH --time=00:30:00             # Wall time limit HH:MM:SS
#SBATCH --output=logs/bhaskera_%j_%N.out   # stdout per node (%j=jobid, %N=nodename)
#SBATCH --error=logs/bhaskera_%j_%N.err    # stderr per node
#SBATCH --exclusive                 # Exclusive node access (no job sharing)

# Optional: email notifications
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email.com

# =============================================================================
# 0. Parse arguments passed to sbatch script
#    Example: sbatch submit_multinode.sh --config config_fsdp.yaml --extra-arg foo
# =============================================================================
EXTRA_ARGS="$@"   # everything passed after the script name is forwarded to bhaskera

# =============================================================================
# 1. Environment Setup — adapt paths to your Rudra environment
# =============================================================================
# Load CUDA via spack (Rudra-specific)
. /home/apps/SPACK/spack/share/spack/setup-env.sh
spack load /lvol4vd      # <-- replace hash with your spack CUDA hash if different

# Activate your Python/conda environment
# Option A — conda
# source /scratch/ldls-iiitd/miniconda3/etc/profile.d/conda.sh
# conda activate bhaskera

# Option B — venv (most portable)
source /scratch/ldls-iiitd/training-framework/Bhaskera/.venv/bin/activate   # EDIT THIS PATH

# Ensure the bhaskera package is importable
export PYTHONPATH="/scratch/ldls-iiitd/training-framework:$PYTHONPATH"

# =============================================================================
# 2. NCCL / networking tuning for InfiniBand (common on Rudra-class HPCs)
#    Comment out the IB lines if your nodes use Ethernet instead.
# =============================================================================
export NCCL_DEBUG=INFO                  # Set to WARN for less verbose logs
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0               # 0 = use InfiniBand; 1 = disable
export NCCL_IB_GID_INDEX=3             # GID index for RoCE v2 (common on Mellanox)
export NCCL_SOCKET_IFNAME=ib0          # InfiniBand interface name; change to eth0 for ethernet
export NCCL_ASYNC_ERROR_HANDLING=1     # Surface NCCL errors properly in Python
export NCCL_TIMEOUT=1800               # 30 min collective timeout (large models need this)

# Prevent CUDA from pre-allocating all GPU memory; lets FSDP manage it
export CUDA_DEVICE_MAX_CONNECTIONS=1

# =============================================================================
# 3. Derive distributed training parameters from SLURM environment
#
#    SLURM sets these automatically once the job starts:
#      SLURM_NNODES         — total number of nodes
#      SLURM_NTASKS_PER_NODE — tasks (GPUs) per node
#      SLURM_NODEID         — 0-indexed node number
#      SLURM_NODELIST       — comma-separated list of nodes, e.g. "gpu[01-02]"
#      SLURM_JOB_NODELIST   — same as above
#
#    We compute:
#      MASTER_ADDR  — hostname of node 0 (all ranks connect here)
#      MASTER_PORT  — any free port; we hash the job ID to avoid collisions
#      WORLD_SIZE   — total number of GPU processes across all nodes
# =============================================================================

# Extract head node hostname (first node in the allocation)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR

# Pick a deterministic but collision-resistant port from the job ID
# Ports 20000-65000 are usually free on HPC login/compute nodes
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 45000) ))
export MASTER_PORT

# Total world size = nodes × GPUs-per-node
WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
export WORLD_SIZE

echo "========================================"
echo "  Bhaskera Multi-Node Training"
echo "========================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Nodes       : $SLURM_NNODES"
echo "  GPUs/node   : $SLURM_NTASKS_PER_NODE"
echo "  World size  : $WORLD_SIZE"
echo "  Master      : $MASTER_ADDR:$MASTER_PORT"
echo "  Node list   : $SLURM_JOB_NODELIST"
echo "  Args        : $EXTRA_ARGS"
echo "========================================"

# Create log directory (must exist before srun)
mkdir -p logs

# =============================================================================
# 4. Launch with srun
#
#    srun spawns exactly one process per task (one task = one GPU).
#    The bhaskera-multinode entrypoint reads SLURM env vars and
#    bootstraps torch.distributed without needing torchrun / Ray.
#
#    --label         — prefix each output line with "node:task" for easy grep
#    --kill-on-bad-exit 1 — kill all tasks if any single task exits non-zero
# =============================================================================
srun \
    --label \
    --kill-on-bad-exit=1 \
    python -m bhaskera.launcher.slurm_entry \
        $EXTRA_ARGS