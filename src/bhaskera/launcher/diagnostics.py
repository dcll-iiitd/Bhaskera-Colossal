"""
bhaskera.launcher.diagnostics
==============================
Run via:  srun python -m bhaskera.launcher.diagnostics

Checks performed on every rank:
  1. SLURM env vars are present and consistent
  2. torch.distributed can be initialised (NCCL backend)
  3. All-reduce across all ranks completes (verifies inter-node comm)
  4. CUDA device is accessible and correct
  5. bfloat16 tensor can be sent cross-node (catches dtype issues)
  6. (Rank 0 only) prints a summary table

Exit code 0 = everything OK.  Non-zero = something is broken.
"""
from __future__ import annotations

import os
import socket
import sys
import time

import torch
import torch.distributed as dist


def _env(key: str, default: str = "MISSING") -> str:
    return os.environ.get(key, default)


def main() -> None:
    global_rank = int(_env("SLURM_PROCID", "0"))
    local_rank  = int(_env("SLURM_LOCALID", "0"))
    world_size  = int(_env("SLURM_NTASKS", "1"))
    hostname    = socket.gethostname()

    master_addr = _env("MASTER_ADDR")
    master_port = _env("MASTER_PORT")

    # ------------------------------------------------------------------ #
    # 1. Basic env check
    # ------------------------------------------------------------------ #
    missing = [k for k in ("MASTER_ADDR", "MASTER_PORT") if _env(k) == "MISSING"]
    if missing:
        print(f"[rank {global_rank}] ERROR: missing env vars: {missing}", flush=True)
        sys.exit(1)

    print(
        f"[rank {global_rank}/{world_size}] host={hostname} "
        f"local_rank={local_rank} master={master_addr}:{master_port}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 2. torch.distributed init
    # ------------------------------------------------------------------ #
    os.environ.update({
        "RANK":        str(global_rank),
        "WORLD_SIZE":  str(world_size),
        "LOCAL_RANK":  str(local_rank),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": master_port,
    })

    t0 = time.time()
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=global_rank, world_size=world_size)
    init_ms = (time.time() - t0) * 1000
    print(f"[rank {global_rank}] dist.init OK in {init_ms:.0f} ms", flush=True)

    # ------------------------------------------------------------------ #
    # 3. CUDA device
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        print(f"[rank {global_rank}] ERROR: CUDA not available!", flush=True)
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    props  = torch.cuda.get_device_properties(device)
    print(
        f"[rank {global_rank}] GPU: {props.name} "
        f"({props.total_memory // 1024**3} GB VRAM)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 4. All-reduce sanity check (float32)
    # ------------------------------------------------------------------ #
    t = torch.ones(1, device=device) * global_rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    assert t.item() == expected, (
        f"[rank {global_rank}] all_reduce mismatch: got {t.item()}, expected {expected}"
    )
    print(f"[rank {global_rank}] all_reduce float32 OK", flush=True)

    # ------------------------------------------------------------------ #
    # 5. All-reduce with bfloat16 (the actual training dtype)
    # ------------------------------------------------------------------ #
    bf = torch.ones(1, device=device, dtype=torch.bfloat16)
    dist.all_reduce(bf, op=dist.ReduceOp.SUM)
    assert abs(bf.item() - world_size) < 0.1, (
        f"[rank {global_rank}] bfloat16 all_reduce failed: got {bf.item()}"
    )
    print(f"[rank {global_rank}] all_reduce bfloat16 OK", flush=True)

    # ------------------------------------------------------------------ #
    # 6. Bandwidth micro-benchmark (optional, ~100 MB)
    # ------------------------------------------------------------------ #
    size = 25 * 1024 * 1024   # 25M floats = 100 MB
    big  = torch.ones(size, device=device)
    dist.barrier()
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    # All-reduce bandwidth formula: 2 × (N-1)/N × size × dtype_bytes / time
    bw_gb = (2 * (world_size - 1) / world_size * size * 4 * 5) / elapsed / 1e9

    if global_rank == 0:
        print(
            f"\n{'='*55}\n"
            f"  Bhaskera multi-node diagnostics PASSED\n"
            f"  Nodes        : {len(set()) or int(_env('SLURM_NNODES','?'))}\n"
            f"  World size   : {world_size}\n"
            f"  NCCL init    : {init_ms:.0f} ms\n"
            f"  Allreduce BW : {bw_gb:.2f} GB/s (est, 100 MB × 5 iters)\n"
            f"{'='*55}\n",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {global_rank}] diagnostics complete", flush=True)


if __name__ == "__main__":
    main()