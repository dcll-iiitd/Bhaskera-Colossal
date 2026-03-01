"""
bhaskera.launcher.ray_entry
============================
Ray Train entry point for multi-node training on SLURM.

Key fixes vs previous version:
  - ray.init(address="auto") instead of ray.init(address="host:port")
    "auto" finds the Ray cluster started on the current node.
  - _wait_for_ray_head() polls the GCS TCP port before connecting,
    replacing the unreliable time.sleep(5).
  - Ray head stdout/stderr redirected to logs/ for debugging.
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
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_ray_head(host: str, port: int, timeout: int = 120) -> None:
    """
    Poll until the Ray head's GCS port accepts TCP connections.

    Replaces time.sleep() — we don't guess how long startup takes,
    we just check until it's actually ready.
    """
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=3):
                elapsed = time.time() - (deadline - timeout)
                logger.info(f"Ray head at {host}:{port} ready (attempt {attempt}, {elapsed:.1f}s)")
                return
        except OSError:
            logger.info(f"Waiting for Ray head at {host}:{port} (attempt {attempt})...")
            time.sleep(3)

    raise TimeoutError(
        f"Ray head at {host}:{port} did not become ready within {timeout}s. "
        "Check logs/ray_head_<jobid>.err for details."
    )


# ---------------------------------------------------------------------------
# Ray cluster bootstrap on SLURM
# ---------------------------------------------------------------------------

def _bootstrap_ray_on_slurm(ray_port: int) -> None:
    """
    Start a Ray cluster using the current SLURM allocation.

    Strategy:
      1. Start Ray head on THIS node (background subprocess).
      2. Poll until GCS port is accepting connections.
      3. Start Ray workers on other nodes via srun --relative=1.
      4. Return — caller uses ray.init(address="auto") to connect.
    """
    head_node     = socket.gethostname()
    num_nodes     = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))
    gpus_per_node = torch.cuda.device_count()
    redis_pass    = os.environ.get("RAY_REDIS_PASSWORD", "")
    job_id        = os.environ.get("SLURM_JOB_ID", "local")
    dashboard_port = int(os.environ.get("RAY_DASHBOARD_PORT", 8265))

    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Start Ray head on this node
    # ------------------------------------------------------------------
    head_cmd = [
        "ray", "start",
        "--head",
        f"--node-ip-address={head_node}",
        f"--port={ray_port}",
        f"--num-gpus={gpus_per_node}",
        f"--num-cpus={cpus_per_node}",
        "--dashboard-host=0.0.0.0",
        f"--dashboard-port={dashboard_port}",
        "--disable-usage-stats",
        "--block",
    ]
    if redis_pass:
        head_cmd += ["--redis-password", redis_pass]

    logger.info(f"Starting Ray head: {head_node}:{ray_port} ({gpus_per_node} GPUs/node)")
    subprocess.Popen(
        head_cmd,
        stdout=open(f"logs/ray_head_{job_id}.out", "w"),
        stderr=open(f"logs/ray_head_{job_id}.err", "w"),
    )

    # ------------------------------------------------------------------
    # 2. Wait for head to be truly ready
    # ------------------------------------------------------------------
    _wait_for_ray_head(head_node, ray_port, timeout=120)

    # ------------------------------------------------------------------
    # 3. Start Ray workers on remaining nodes
    # ------------------------------------------------------------------
    if num_nodes > 1:
        worker_cmd_str = (
            f"ray start "
            f"--address={head_node}:{ray_port} "
            f"--num-gpus={gpus_per_node} "
            f"--num-cpus={cpus_per_node} "
            "--block"
        )
        if redis_pass:
            worker_cmd_str += f" --redis-password {redis_pass}"

        srun_cmd = [
            "srun",
            "--relative=1",
            f"--ntasks={num_nodes - 1}",
            "--ntasks-per-node=1",
            "bash", "-c", worker_cmd_str,
        ]
        logger.info(f"Starting Ray workers on {num_nodes - 1} additional node(s)")
        subprocess.Popen(
            srun_cmd,
            stdout=open(f"logs/ray_workers_{job_id}.out", "w"),
            stderr=open(f"logs/ray_workers_{job_id}.err", "w"),
        )

        # Brief wait for workers to register — head is up so ray.init("auto")
        # won't fail, but we want workers visible before TorchTrainer starts
        logger.info("Waiting for Ray workers to register...")
        time.sleep(15)

    logger.info(
        f"Ray cluster bootstrapped | head={head_node}:{ray_port} "
        f"| dashboard={head_node}:{dashboard_port} "
        f"(SSH tunnel: ssh -L {dashboard_port}:{head_node}:{dashboard_port} <login_node>)"
    )


# ---------------------------------------------------------------------------
# Ray init
# ---------------------------------------------------------------------------

def _init_ray(num_workers: int) -> None:
    """
    Connect to Ray, bootstrapping on SLURM if needed.

    Priority:
      1. RAY_ADDRESS env var set  → connect to external cluster
      2. Multi-node SLURM job    → bootstrap ourselves, connect via "auto"
      3. Single-node / local     → plain ray.init()
    """
    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized, reusing")
        return

    # Derive port from job ID to avoid collisions between concurrent jobs
    ray_port = int(os.environ.get(
        "RAY_PORT",
        6379 + (int(os.environ.get("SLURM_JOB_ID", 0)) % 1000)
    ))

    existing  = os.environ.get("RAY_ADDRESS")
    in_slurm  = "SLURM_JOB_ID" in os.environ
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))

    if existing:
        logger.info(f"Connecting to existing Ray cluster at {existing}")
        ray.init(address=existing, ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    elif in_slurm and num_nodes > 1:
        # Bootstrap Ray on SLURM, then connect via "auto"
        # "auto" = find the Ray cluster running on this node
        _bootstrap_ray_on_slurm(ray_port)
        logger.info("Connecting to Ray cluster via auto-discovery")
        ray.init(address="auto", ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    else:
        logger.info(f"Starting local Ray instance ({num_workers} GPUs)")
        ray.init(num_gpus=num_workers, ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    resources    = ray.available_resources()
    visible_gpus = int(resources.get("GPU", 0))
    logger.info(f"Ray online | GPUs={visible_gpus}/{num_workers} | nodes={num_nodes}")

    if visible_gpus < num_workers:
        logger.warning(
            f"Only {visible_gpus}/{num_workers} GPUs visible — "
            "training will block until resources are available."
        )


# ---------------------------------------------------------------------------
# Ray Train worker (runs inside each actor, one per GPU)
# ---------------------------------------------------------------------------

def _ray_train_func(train_loop_config: dict) -> None:
    """
    Called by Ray Train inside each worker actor.
    Ray Train has already called dist.init_process_group() before this runs.
    """
    import ray.train

    ctx_ray     = ray.train.get_context()
    local_rank  = ctx_ray.get_local_rank()
    global_rank = ctx_ray.get_world_rank()
    world_size  = ctx_ray.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

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

    _init_ray(num_workers)

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 2},
    )
    torch_config = TorchConfig(backend="nccl", timeout_s=1800)
    failure_config = FailureConfig(max_failures=args.max_failures)
    checkpoint_config = CheckpointConfig(num_to_keep=3)

    run_name = args.run_name or f"bhaskera_{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_config = RunConfig(
        name=run_name,
        storage_path=os.path.abspath(args.ray_results_dir),
        failure_config=failure_config,
        checkpoint_config=checkpoint_config,
    )

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

    logger.info(f"Launching TorchTrainer | workers={num_workers} | run={run_name}")
    result = trainer.fit()

    if result.error:
        logger.error(f"Training failed: {result.error}")
        raise RuntimeError(result.error)

    logger.info(f"Training complete! Results: {result.path}")
    ray.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera Ray Train entrypoint")
    p.add_argument("--config",          required=True)
    p.add_argument("--num-workers",     type=int, default=1)
    p.add_argument("--max-failures",    type=int, default=2)
    p.add_argument("--ray-results-dir", type=str, default="./ray_results")
    p.add_argument("--run-name",        type=str, default=None)
    _launch(p.parse_args())


if __name__ == "__main__":
    main()