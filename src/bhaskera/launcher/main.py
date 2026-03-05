"""
bhaskera.launcher.main
=======================
The `bhaskera` CLI entry point.

After `pip install -e .` you can run:

    bhaskera --config config.yaml

Everything is driven from the YAML `launcher:` block.
CLI flags override YAML when provided.

YAML launcher block (all fields optional — shown with defaults):

    launcher:
      backend: ray          # ray | torchrun | slurm
      num_workers: 2        # total GPU workers (ray/torchrun)
      num_nodes: 1          # nodes to span (torchrun multi-node)
      node_rank: 0          # this node's rank (torchrun multi-node)
      master_addr: auto     # head node IP/hostname  (auto = detect)
      master_port: 29500    # rendezvous port
      max_failures: 2       # [ray] worker restart attempts
      ray_results_dir: ./ray_results
      run_name: null        # experiment run name

Backends
--------
ray       Single driver process. Bootstraps Ray cluster automatically.
          Works on: laptop, bare-metal, cloud VM, SLURM — no extra setup.

torchrun  Spawns workers via torchrun (subprocess). Works single and
          multi-node without SLURM. On multi-node set num_nodes + master_addr.

slurm     Called by srun (one process per GPU). Reads SLURM env vars.
          Use inside submit_multinode.sh.

Examples
--------
# Simplest — Ray, all GPUs on current machine:
bhaskera --config config_ray.yaml

# Override num workers at CLI:
bhaskera --config config.yaml --num-workers 4

# Torchrun, 2 GPUs — config.yaml has: launcher: {backend: torchrun, num_workers: 2}
bhaskera --config config.yaml

# Multi-node torchrun (run on EACH node, set node_rank per node in YAML or CLI):
bhaskera --config config.yaml --num-workers 4 --num-nodes 2 --node-rank 0 --master-addr 10.0.0.1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][bhaskera] %(levelname)s %(message)s",
)


# ── YAML launcher config reader ───────────────────────────────────────────────

def _read_launcher_yaml(config_path: str) -> Dict[str, Any]:
    """
    Read the `launcher:` block from the YAML config.

    Supports both compact and expanded forms:

        launcher: ray                  # compact string -> {backend: ray}
        launcher:                      # expanded dict
          backend: ray
          num_workers: 4
    """
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not read config '{config_path}': {e}")
        return {}

    raw = data.get("launcher", {})
    if isinstance(raw, str):
        return {"backend": raw.lower()}
    if isinstance(raw, dict):
        result = dict(raw)
        if "backend" in result:
            result["backend"] = str(result["backend"]).lower()
        return result
    return {}


# ── torchrun backend ──────────────────────────────────────────────────────────

def _launch_torchrun(config_path: str, lc: Dict[str, Any]) -> None:
    """
    Spawn torchrun as a subprocess, then exec into it (replaces current process).
    torchrun manages all worker processes from here.
    """
    import torch

    num_workers = int(lc.get("num_workers") or torch.cuda.device_count() or 1)
    num_nodes   = int(lc.get("num_nodes",  1))
    node_rank   = int(lc.get("node_rank",  0))
    master_port = str(lc.get("master_port", 29500))

    master_addr = lc.get("master_addr", "auto")
    if master_addr in (None, "auto", ""):
        import socket
        master_addr = socket.gethostname()

    nproc = max(1, num_workers // num_nodes)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m", "bhaskera.launcher.torchrun_entry",
        "--config", config_path,
    ]

    logger.info(
        f"[torchrun] nodes={num_nodes} procs/node={nproc} "
        f"master={master_addr}:{master_port} node_rank={node_rank}"
    )

    # Replace current process — clean exit code propagation
    os.execvp(sys.executable, cmd)


# ── ray backend ───────────────────────────────────────────────────────────────

def _launch_ray(config_path: str, lc: Dict[str, Any]) -> None:
    import torch
    from bhaskera.launcher.ray_entry import _launch

    num_workers = int(lc.get("num_workers") or torch.cuda.device_count() or 1)

    args = argparse.Namespace(
        config=config_path,
        num_workers=num_workers,
        max_failures=int(lc.get("max_failures", 2)),
        ray_results_dir=str(lc.get("ray_results_dir", "./ray_results")),
        run_name=lc.get("run_name") or None,
    )
    _launch(args)


# ── slurm backend ─────────────────────────────────────────────────────────────

def _launch_slurm(config_path: str) -> None:
    """srun already placed us on the right GPU — just run the worker."""
    from bhaskera.launcher.slurm_entry import _run_worker
    _run_worker(argparse.Namespace(config=config_path))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="bhaskera",
        description="Bhaskera LLM fine-tuning launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
All settings live in the YAML config under the `launcher:` block.
CLI flags always override YAML values.

Minimal YAML:
  launcher: ray            # just pick a backend

Full YAML launcher block:
  launcher:
    backend: ray           # ray | torchrun | slurm
    num_workers: 4         # total GPU workers
    num_nodes: 1           # [torchrun] node count
    node_rank: 0           # [torchrun] rank of this node
    master_addr: auto      # [torchrun] head node IP
    master_port: 29500
    max_failures: 2        # [ray] crash retries
    ray_results_dir: ./ray_results
    run_name: my-run

Examples:
  bhaskera --config config.yaml
  bhaskera --config config.yaml --num-workers 4
  bhaskera --config config.yaml --backend torchrun --num-workers 2
        """,
    )

    p.add_argument("--config",       required=True,
                   help="Path to YAML config file")
    p.add_argument("--backend",      choices=["ray", "torchrun", "slurm"], default=None,
                   help="Override launcher backend from YAML")
    p.add_argument("--num-workers",  type=int,  default=None,
                   help="Total GPU workers (overrides YAML launcher.num_workers)")
    p.add_argument("--num-nodes",    type=int,  default=None,
                   help="[torchrun] Number of nodes")
    p.add_argument("--node-rank",    type=int,  default=None,
                   help="[torchrun] Rank of this node (0-indexed)")
    p.add_argument("--master-addr",  type=str,  default=None,
                   help="[torchrun] Head node hostname or IP")
    p.add_argument("--master-port",  type=int,  default=None,
                   help="Rendezvous port")
    p.add_argument("--max-failures", type=int,  default=None,
                   help="[ray] Max worker restart attempts")
    p.add_argument("--run-name",     type=str,  default=None,
                   help="Experiment run name")

    args = p.parse_args()

    # Load launcher settings from YAML
    lc = _read_launcher_yaml(args.config)

    # Apply CLI overrides (CLI always wins)
    if args.backend      is not None: lc["backend"]      = args.backend
    if args.num_workers  is not None: lc["num_workers"]  = args.num_workers
    if args.num_nodes    is not None: lc["num_nodes"]    = args.num_nodes
    if args.node_rank    is not None: lc["node_rank"]    = args.node_rank
    if args.master_addr  is not None: lc["master_addr"]  = args.master_addr
    if args.master_port  is not None: lc["master_port"]  = args.master_port
    if args.max_failures is not None: lc["max_failures"] = args.max_failures
    if args.run_name     is not None: lc["run_name"]     = args.run_name

    backend = lc.get("backend", "ray").lower()

    logger.info(f"Backend : {backend}")
    logger.info(f"Config  : {args.config}")
    for k, v in lc.items():
        if k != "backend":
            logger.info(f"  {k}: {v}")

    if backend == "ray":
        _launch_ray(args.config, lc)
    elif backend == "torchrun":
        _launch_torchrun(args.config, lc)
    elif backend == "slurm":
        _launch_slurm(args.config)
    else:
        logger.error(f"Unknown backend: '{backend}'. Choose: ray, torchrun, slurm")
        sys.exit(1)


if __name__ == "__main__":
    main()
