"""
bhaskera.launcher.ray_entry
============================
Ray Train entry point for multi-node training on SLURM.

Architecture on Rudra (SLURM-only cluster)
-------------------------------------------
Ray cannot run as a persistent daemon on a SLURM cluster.  Instead we use
the "Ray-on-SLURM" pattern:

  1. SLURM allocates N nodes.
  2. This script runs on node 0 (the Ray head).
  3. It SSHes (via `srun --relative`) to start Ray workers on other nodes.
  4. ray.init() connects to the just-started cluster.
  5. TorchTrainer spawns one Ray actor per GPU.
  6. Each actor calls worker_core.run_worker() — same code as the SLURM path.

What Ray adds over the raw SLURM path
--------------------------------------
  Fault tolerance     — TorchTrainer automatically restarts failed workers up
                        to `max_failures` times without resubmitting the job.
  Dynamic scaling     — Ray can add/remove workers mid-run if the cluster
                        has spare capacity (not typical on SLURM, but supported).
  Unified tracking    — Ray Train reports metrics to the Ray dashboard AND
                        your existing MLflow/W&B logger simultaneously.
  Checkpoint resume   — Ray Train has a built-in checkpoint API that integrates
                        with your existing save_checkpoint() calls.

Usage (via submit_ray.sh):
    python -m bhaskera.launcher.ray_entry --config config_ray.yaml [--num-workers 4]

Usage (local, single node):
    python -m bhaskera.launcher.ray_entry --config config_ray.yaml --num-workers 2
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import time

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][ray_entry] %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Ray cluster bootstrap on SLURM
# ---------------------------------------------------------------------------

def _bootstrap_ray_on_slurm() -> str:
    """
    Start a Ray cluster using the SLURM allocation.

    Returns the Ray address string to pass to ray.init().

    Strategy:
      - rank 0 starts the head: `ray start --head --port=6379`
      - ranks 1..N start workers: `ray start --address=<head>:6379`
      - We use `srun --relative` to target specific nodes without a new job.
    """
    import ray

    head_node  = socket.gethostname()
    ray_port   = int(os.environ.get("RAY_PORT", 6379))
    redis_pass = os.environ.get("RAY_REDIS_PASSWORD", "")

    num_nodes  = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_task = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))

    # ------------------------------------------------------------------
    # Start Ray head on this node
    # ------------------------------------------------------------------
    head_cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={head_node}",
        f"--port={ray_port}",
        "--num-gpus", str(torch.cuda.device_count()),
        "--num-cpus", str(cpus_per_task),
        "--block",   # keep process alive; we'll manage lifetime ourselves
    ]
    if redis_pass:
        head_cmd += ["--redis-password", redis_pass]

    logger.info(f"Starting Ray head on {head_node}:{ray_port}")
    # Run in background — block=True keeps the Ray head alive
    head_proc = subprocess.Popen(
        head_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Give the head a moment to start
    time.sleep(5)

    # ------------------------------------------------------------------
    # Start Ray workers on the other nodes
    # ------------------------------------------------------------------
    if num_nodes > 1:
        worker_cmd = (
            f"ray start "
            f"--address={head_node}:{ray_port} "
            f"--num-gpus={torch.cuda.device_count()} "
            f"--num-cpus={cpus_per_task} "
            f"--block"
        )
        if redis_pass:
            worker_cmd += f" --redis-password {redis_pass}"

        # srun --relative=1 targets nodes 1..N-1 (skipping node 0 = head)
        srun_cmd = [
            "srun",
            "--relative=1",
            f"--ntasks={num_nodes - 1}",
            "--ntasks-per-node=1",
            "bash", "-c", worker_cmd,
        ]
        logger.info(f"Starting Ray workers on {num_nodes - 1} additional node(s)")
        subprocess.Popen(
            srun_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for workers to register
        time.sleep(10)

    ray_address = f"ray://{head_node}:{ray_port}"
    logger.info(f"Ray cluster ready at {ray_address}")
    return f"{head_node}:{ray_port}"   # format for ray.init(address=...)


def _init_ray(num_workers: int) -> None:
    """
    Connect to Ray — either bootstrapping on SLURM or plain ray.init().
    """
    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized, reusing existing cluster")
        return

    in_slurm = "SLURM_JOB_ID" in os.environ
    existing  = os.environ.get("RAY_ADDRESS")   # e.g. set by ray job submit

    if existing:
        # Connecting to an externally started Ray cluster
        logger.info(f"Connecting to existing Ray cluster at {existing}")
        ray.init(address=existing, ignore_reinit_error=True)

    elif in_slurm and int(os.environ.get("SLURM_NNODES", 1)) > 1:
        # Multi-node SLURM: bootstrap Ray ourselves
        address = _bootstrap_ray_on_slurm()
        ray.init(address=address, ignore_reinit_error=True)

    else:
        # Single-node or local dev: plain ray.init()
        logger.info(f"Starting local Ray instance with {num_workers} GPUs")
        ray.init(
            num_gpus=num_workers,
            ignore_reinit_error=True,
        )

    # Verify resources
    resources = ray.available_resources()
    visible_gpus = resources.get("GPU", 0)
    logger.info(f"Ray sees {visible_gpus} GPUs (need {num_workers})")
    if visible_gpus < num_workers:
        logger.warning(
            f"Ray only sees {visible_gpus} GPUs but {num_workers} workers "
            "requested. Training will block waiting for resources."
        )


# ---------------------------------------------------------------------------
# Ray Train worker function
# ---------------------------------------------------------------------------

def _ray_train_func(train_loop_config: dict) -> None:
    """
    Runs inside each Ray Train worker (one per GPU).

    Ray Train has already called dist.init_process_group() before this
    function runs, so we just read the context and hand off to worker_core.
    """
    import ray.train
    import torch.distributed as dist

    ctx_ray    = ray.train.get_context()
    local_rank  = ctx_ray.get_local_rank()
    global_rank = ctx_ray.get_world_rank()
    world_size  = ctx_ray.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load config inside the worker (workers may be on different nodes)
    config_path = train_loop_config["config_path"]
    submit_dir  = train_loop_config.get("submit_dir", ".")
    os.chdir(submit_dir)

    from bhaskera.config_loader import load_config
    cfg = load_config(config_path)

    from bhaskera.launcher.worker_core import WorkerContext, run_worker
    ctx = WorkerContext(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        launcher="ray",
    )
    run_worker(ctx, cfg)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

def _launch(args: argparse.Namespace) -> None:
    import ray
    from ray.train.torch import TorchTrainer, TorchConfig
    from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig

    num_workers = args.num_workers

    # ---- initialise Ray (handles SLURM bootstrap automatically) ----------
    _init_ray(num_workers)

    # ---- Ray Train config ------------------------------------------------
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 2},
    )

    torch_config = TorchConfig(
        backend="nccl",
        timeout_s=1800,
    )

    # Fault tolerance: retry failed workers up to max_failures times
    # Set max_failures=0 to disable restarts (same behaviour as SLURM path)
    failure_config = FailureConfig(
        max_failures=args.max_failures,
    )

    # Ray Train checkpoint integration — works alongside your existing
    # save_checkpoint() calls; Ray can resume from these automatically
    checkpoint_config = CheckpointConfig(
        num_to_keep=3,                      # mirrors CHECKPOINT_KEEP_LAST_N
    )

    run_config = RunConfig(
        name=args.run_name or f"bhaskera_{os.environ.get('SLURM_JOB_ID', 'local')}",
        storage_path=args.ray_results_dir,
        failure_config=failure_config,
        checkpoint_config=checkpoint_config,
    )

    # ---- TorchTrainer ----------------------------------------------------
    trainer = TorchTrainer(
        train_loop_per_worker=_ray_train_func,
        train_loop_config={
            "config_path": args.config,
            "submit_dir":  os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
        },
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )

    logger.info(f"Launching Ray TorchTrainer with {num_workers} workers")
    result = trainer.fit()

    if result.error:
        logger.error(f"Training failed: {result.error}")
        raise RuntimeError(result.error)
    else:
        logger.info("Training complete!")
        logger.info(f"Ray results dir: {result.path}")

    ray.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera Ray Train entrypoint")
    p.add_argument("--config",          required=True,        help="YAML config path")
    p.add_argument("--num-workers",     type=int, default=1,  help="Number of GPUs / Ray workers")
    p.add_argument("--max-failures",    type=int, default=2,  help="Ray fault tolerance: max worker restarts (0=off)")
    p.add_argument("--ray-results-dir", type=str, default="./ray_results", help="Ray Train results directory")
    p.add_argument("--run-name",        type=str, default=None, help="Ray run name (defaults to job ID)")
    _launch(p.parse_args())


if __name__ == "__main__":
    main()