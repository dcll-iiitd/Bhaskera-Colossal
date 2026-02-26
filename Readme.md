# Bhaskera — Multi-Node Training & SLURM Integration

## What was added

```
Bhaskera/
├── config_multinode.yaml                   # Multi-node training config
├── slurm/
│   ├── submit_multinode.sh                 # Main submission script (EDIT paths here)
│   ├── sweep_array.sh                      # Array job for HP sweeps
│   └── diagnose_multinode.sh               # Pre-flight connectivity check
└── src/bhaskera/launcher/
    ├── __init__.py
    ├── slurm_entry.py                      # srun worker (no Ray, no torchrun)
    ├── torchrun_entry.py                   # torchrun worker (local dev / testing)
    └── diagnostics.py                      # NCCL + bandwidth diagnostic
```

The existing `cli.py` (Ray-based) and `newtrain.py` are **unchanged**. The
new SLURM path is a parallel launch route that shares 100% of the training
logic (`train_loop.py`, `wrapper.py`, `fsdp_utils.py`, etc.).

---

## Architecture: how the three launch paths relate

```
                        ┌─────────────────────────────────────┐
                        │         shared training code         │
                        │  train_loop.py · fsdp_utils.py       │
                        │  wrapper.py · models/ · data/        │
                        └────────────┬────────────┬────────────┘
                                     │            │
              ┌──────────────────────┤            ├──────────────────────┐
              ▼                      ▼            ▼                      ▼
   launcher/slurm_entry.py    cli.py (Ray)  newtrain.py (Ray)  launcher/torchrun_entry.py
   ─────────────────────────  ────────────  ─────────────────  ──────────────────────────
   srun → one process/GPU     Ray workers   Ray workers        torchrun → one process/GPU
   SLURM env vars for dist    Ray handles   Ray handles        torchrun env vars for dist
   Best for: Rudra HPC        Best for:     Best for:          Best for: local dev /
   multi-node production      cloud / k8s   cloud / k8s        single-node debug
```

---

## Quickstart on Rudra HPC

### Step 1: Edit paths in the submission script

Open `slurm/submit_multinode.sh` and update the two environment lines:

```bash
spack load /lvol4vd                                       # your spack CUDA hash
source /scratch/ldls-iiitd/training-framework/venv/bin/activate  # your venv path
```

### Step 2: Check your network interface

SSH into a compute node and run `ip link show` or `ibstat`. If you have
InfiniBand, `ib0` is usually correct. If Ethernet only, change:
```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0    # or the right ethernet interface
```

### Step 3: Run the diagnostic first (always!)

```bash
mkdir -p logs
sbatch slurm/diagnose_multinode.sh
# Watch: tail -f logs/diag_<jobid>_*.out
```

You should see:
```
  Bhaskera multi-node diagnostics PASSED
  World size   : 4
  NCCL init    : 1234 ms
  Allreduce BW : 12.34 GB/s (est, 100 MB × 5 iters)
```

### Step 4: Submit training

```bash
sbatch slurm/submit_multinode.sh --config config_multinode.yaml
```

Override node count at submission time:
```bash
sbatch --nodes=4 --ntasks-per-node=4 --gres=gpu:4 \
    slurm/submit_multinode.sh --config config_multinode.yaml
```

### Step 5: Monitor

```bash
squeue -u $USER                        # job status
tail -f logs/bhaskera_<jobid>_*.out    # live logs from all nodes
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed   # after job ends
```

---

## How the SLURM launch works (key concepts)

### `srun` vs `torchrun` vs Ray

| | `srun` (our path) | `torchrun` | Ray |
|---|---|---|---|
| Process management | SLURM | PyTorch launcher | Ray cluster |
| Fault tolerance | SLURM job-level | Elastic (optional) | Actor restarts |
| Multi-node | Native | Needs manual addr/port | Ray cluster |
| Best for | HPC batch jobs | Dev / single-node | Cloud / k8s |

### How `srun` sets up ranks

SLURM sets these automatically per task:
```
SLURM_PROCID   → global rank  (0 … world_size-1)
SLURM_LOCALID  → local rank   (0 … gpus_per_node-1)
SLURM_NTASKS   → world size
```

`slurm_entry.py` reads these and calls `dist.init_process_group(init_method="env://")`.
No `torchrun`, no Ray, no manual rendezvous — just SLURM + NCCL.

### Why `MASTER_ADDR`/`MASTER_PORT` are computed in the script

```bash
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$(( 20000 + (SLURM_JOB_ID % 45000) ))
```

- `scontrol show hostnames` expands compact node lists like `gpu[01-02]` → `gpu01 gpu02`
- The port is derived from the job ID so concurrent jobs don't collide

### FSDP + multi-node: what changes vs single-node

Nothing in `fsdp_utils.py` or `wrapper.py` changes. FSDP's sharding is
topology-agnostic — it shards across `world_size` processes regardless of
whether they're on one node or ten. The only things that change are:

1. More processes → each holds a smaller shard → less GPU memory per GPU
2. Inter-node all-gathers are slower than NVLink → `BACKWARD_PRE` prefetch matters more
3. `NCCL_IB_*` settings must match your fabric

---

## NCCL tuning reference for Rudra

```bash
# InfiniBand (most likely on Rudra):
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3        # RoCE v2; try 1 if 3 doesn't work
export NCCL_SOCKET_IFNAME=ib0     # IB interface name

# Ethernet fallback:
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Always set:
export NCCL_ASYNC_ERROR_HANDLING=1   # surface errors properly
export NCCL_TIMEOUT=1800             # 30 min (large model all-gathers are slow)
export NCCL_DEBUG=INFO               # switch to WARN once stable
```

---

## Fixing the existing `newtrain.py` bug

There is a `NameError` in `newtrain.py` around the logger setup block. The variable
is named `cfg` throughout, but the logger block accidentally uses `config`:

```python
# BROKEN (newtrain.py ~line 110):
if config.TRACKER and global_rank == 0:
    log_gpu = getattr(config, "log_gpu", True)

# FIXED:
if cfg.TRACKER and global_rank == 0:
    log_gpu = getattr(cfg, "log_gpu", True)
```

The SLURM path (`slurm_entry.py`) does not have this bug. Fix `newtrain.py` if
you want the Ray path to work with the new YAML config too.

---

## Hyperparameter sweep (array jobs)

Edit `slurm/sweep_array.sh` to list your configs:
```bash
CONFIGS=(
    "config_lr1e4.yaml"
    "config_lr5e5.yaml"
    "config_r32.yaml"
)
```

Then:
```bash
sbatch slurm/sweep_array.sh            # submits all 3
sbatch --array=0-1 slurm/sweep_array.sh  # only first two
```

Each array task gets a different port (`MASTER_PORT + SLURM_ARRAY_TASK_ID`)
so they don't collide.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Job hangs at `dist.init_process_group` | MASTER_ADDR unreachable | Check `NCCL_SOCKET_IFNAME`, firewall |
| NCCL timeout after `dist.barrier()` | One rank crashed silently | Check `.err` logs on all nodes |
| `Address already in use` | Port collision | Increase port range or use `--exclusive` |
| NaN loss from step 1 | LR too high or no warmup | Use `warmup_steps: 20` in config |
| FSDP hangs during checkpoint | Rank 0 calling save alone | Already fixed in `train_loop.py` — verify all ranks reach checkpoint block |
| `ibstat: command not found` | IB tools not loaded | `module load infiniband-diags` or check spack |