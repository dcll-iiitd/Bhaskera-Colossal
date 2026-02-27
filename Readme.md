# Bhaskera — Dual Backend: SLURM + Ray

## File overview

```
Bhaskera/
├── config_ray.yaml                        ← Ray backend config (has launcher: ray)
├── config_multinode.yaml                  ← SLURM backend config (existing)
│
├── slurm/
│   ├── submit_multinode.sh                ← SLURM backend submit (existing)
│   └── submit_ray.sh                      ← Ray backend submit (NEW)
│
└── src/bhaskera/launcher/
    ├── __init__.py
    ├── main.py           ← Unified CLI: --launcher flag selects backend  (NEW)
    ├── worker_core.py    ← Shared training logic called by both backends  (NEW)
    ├── slurm_entry.py    ← SLURM backend worker (refactored, uses worker_core)
    ├── ray_entry.py      ← Ray backend driver + worker (NEW)
    ├── torchrun_entry.py ← Local dev (existing)
    └── diagnostics.py   ← NCCL diagnostics (existing)
```

---

## How to use

### SLURM backend (current working path)

```bash
sbatch slurm/submit_multinode.sh --config config_multinode.yaml
```

Or explicitly via the unified CLI (still called by srun):
```bash
# submit_multinode.sh calls:
srun python -m bhaskera.launcher.main --launcher slurm --config config_multinode.yaml
```

### Ray backend (new)

```bash
sbatch slurm/submit_ray.sh --config config_ray.yaml
```

Or locally with 2 GPUs:
```bash
python -m bhaskera.launcher.main --launcher ray --config config_ray.yaml --num-workers 2
```

### Backend selection priority

```
--launcher flag  >  YAML launcher field  >  default: slurm
```

This means `config_ray.yaml` has `launcher: ray` but you can always override:
```bash
# Force SLURM backend even though config says ray:
srun python -m bhaskera.launcher.main --launcher slurm --config config_ray.yaml
```

---

## Architecture: how the two backends share code

```
submit_multinode.sh          submit_ray.sh
       │                           │
       │ srun (one proc/GPU)       │ python (one proc total, head node)
       ▼                           ▼
slurm_entry.py             ray_entry.py
  - reads SLURM env vars      - bootstraps Ray cluster via srun --relative
  - dist.init_process_group   - calls ray.init()
  - creates WorkerContext      - TorchTrainer spawns workers
       │                           │ (Ray calls dist.init_process_group)
       │                           │ creates WorkerContext
       └──────────┬────────────────┘
                  ▼
          worker_core.run_worker(ctx, cfg)
          ─────────────────────────────────
          tokenizer → dataset → model
          → wrap_model_distributed (DDP/FSDP)
          → optimizer → train_loop
```

---

## What Ray adds over the raw SLURM path

| Feature | SLURM path | Ray path |
|---|---|---|
| Process management | SLURM (srun) | Ray actors |
| Worker fault tolerance | ❌ job fails on any crash | ✅ restarts up to `--max-failures` times |
| Dynamic scaling | ❌ | ✅ (if spare SLURM allocation exists) |
| Unified dashboard | ❌ | ✅ Ray dashboard on `head:8265` |
| Checkpoint resume | manual | ✅ Ray Train CheckpointConfig |
| Training code changes | none | none (same worker_core) |

---

## Ray-on-SLURM: how the bootstrap works

On a SLURM-only cluster like Rudra, there is no persistent Ray cluster.
`ray_entry.py` bootstraps one automatically:

```
submit_ray.sh
  │
  └── python -m bhaskera.launcher.main --launcher ray
            │
            ├── ray start --head --port=6379   (on rpgpu005, background)
            │
            ├── srun --relative=1 --ntasks=1   (on rpgpu006)
            │     ray start --address=rpgpu005:6379
            │
            └── ray.init(address="rpgpu005:6379")
                      │
                      └── TorchTrainer.fit()
                                │
                                ├── Ray actor on rpgpu005 GPU 0
                                ├── Ray actor on rpgpu005 GPU 1
                                ├── Ray actor on rpgpu006 GPU 0
                                └── Ray actor on rpgpu006 GPU 1
```

---

## Fault tolerance in practice

With `--max-failures 2` (default), if any GPU worker crashes mid-training:

1. Ray detects the failure
2. Ray restarts the failed worker (up to 2 times)
3. Training resumes from the last checkpoint saved by `train_loop.py`
4. If failures exceed the limit, the job exits with an error

To disable restarts (same as SLURM path behaviour):
```bash
sbatch slurm/submit_ray.sh --config config_ray.yaml --max-failures 0
```

---

## The `1e8` bug fix

In `config_loader.py` line 129, change:
```python
# BROKEN — YAML parses 1e8 as the string '1e8', int() can't parse it
fsdp_min_num_params=int(auto_wrap.get('min_num_params', 1e8)),

# FIXED
fsdp_min_num_params=int(float(auto_wrap.get('min_num_params', 1e8))),
```

Or just use a plain integer in your YAML (already done in config_ray.yaml):
```yaml
min_num_params: 100000000   # avoids the issue entirely
```