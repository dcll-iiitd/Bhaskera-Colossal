"""
bhaskera.launcher.diagnostics
==============================
Run before a full training job to verify multi-node setup.

Usage:
    # SLURM:
    srun python -m bhaskera.launcher.diagnostics

    # torchrun:
    torchrun --nnodes=2 --nproc_per_node=2 ... -m bhaskera.launcher.diagnostics
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
    # Support both SLURM and torchrun env var names
    global_rank = int(_env("SLURM_PROCID",   _env("RANK",       "0")))
    local_rank  = int(_env("SLURM_LOCALID",  _env("LOCAL_RANK", "0")))
    world_size  = int(_env("SLURM_NTASKS",   _env("WORLD_SIZE", "1")))
    hostname    = socket.gethostname()
    master_addr = _env("MASTER_ADDR")
    master_port = _env("MASTER_PORT")

    missing = [k for k in ("MASTER_ADDR", "MASTER_PORT") if _env(k) == "MISSING"]
    if missing:
        print(f"[rank {global_rank}] ERROR: missing env vars: {missing}", flush=True)
        sys.exit(1)

    print(
        f"[rank {global_rank}/{world_size}] host={hostname} "
        f"local_rank={local_rank} master={master_addr}:{master_port}",
        flush=True,
    )

    os.environ.update({
        "RANK": str(global_rank), "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(local_rank),
        "MASTER_ADDR": master_addr, "MASTER_PORT": master_port,
    })

    t0 = time.time()
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=global_rank, world_size=world_size)
    init_ms = (time.time() - t0) * 1000
    print(f"[rank {global_rank}] dist.init OK in {init_ms:.0f} ms", flush=True)

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

    # All-reduce float32
    t = torch.ones(1, device=device) * global_rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert t.item() == sum(range(world_size)), \
        f"[rank {global_rank}] all_reduce mismatch: {t.item()} != {sum(range(world_size))}"
    print(f"[rank {global_rank}] all_reduce float32 OK", flush=True)

    # All-reduce bfloat16
    bf = torch.ones(1, device=device, dtype=torch.bfloat16)
    dist.all_reduce(bf, op=dist.ReduceOp.SUM)
    assert abs(bf.item() - world_size) < 0.1, \
        f"[rank {global_rank}] bfloat16 all_reduce failed: {bf.item()}"
    print(f"[rank {global_rank}] all_reduce bfloat16 OK", flush=True)

    # Bandwidth benchmark (~100 MB)
    size = 25 * 1024 * 1024
    big  = torch.ones(size, device=device)
    dist.barrier()
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(big, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    bw_gb = (2 * (world_size - 1) / max(world_size, 2) * size * 4 * 5) / elapsed / 1e9

    # Collect all hostnames for the summary
    hostname_tensor = torch.zeros(256, dtype=torch.uint8, device=device)
    for i, c in enumerate(hostname.encode()[:256]):
        hostname_tensor[i] = c
    all_hostnames_t = [torch.zeros(256, dtype=torch.uint8, device=device)
                       for _ in range(world_size)]
    dist.all_gather(all_hostnames_t, hostname_tensor)

    if global_rank == 0:
        nodes = sorted({bytes(t.cpu().tolist()).split(b'\x00')[0].decode()
                        for t in all_hostnames_t})
        print(
            f"\n{'='*60}\n"
            f"  Bhaskera multi-node diagnostics PASSED\n"
            f"  Nodes ({len(nodes)}): {', '.join(nodes)}\n"
            f"  World size   : {world_size}\n"
            f"  NCCL init    : {init_ms:.0f} ms\n"
            f"  Allreduce BW : {bw_gb:.2f} GB/s (est, 100 MB × 5 iters)\n"
            f"{'='*60}\n",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {global_rank}] diagnostics complete", flush=True)


if __name__ == "__main__":
    main()
