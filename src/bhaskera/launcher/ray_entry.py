"""
bhaskera.launcher.ray_entry
============================
Ray Train entry point.  Works in three environments:

  1. Plain local machine (single or multi-GPU)
     python -m bhaskera.launcher.main --launcher ray --config cfg.yaml

  2. Multi-node WITHOUT SLURM (e.g. bare-metal cluster, Kubernetes)
     Set RAY_ADDRESS=<head>:<port> and start Ray head manually, OR
     let this script bootstrap a single-node Ray cluster and specify
     --num-workers to use all local GPUs.

  3. Multi-node WITH SLURM (e.g. Rudra HPC)
     sbatch slurm/submit_ray.sh --config cfg.yaml
     (script bootstraps Ray head+workers using srun --relative)

Ray init priority:
  RAY_ADDRESS env var  →  existing cluster
  SLURM multi-node     →  bootstrap head + workers via srun
  everything else      →  local ray.init()
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
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][ray_entry] %(levelname)s %(message)s",
)


# ── TCP polling helper ─────────────────────────────────────────────────────────

def _wait_for_tcp(host: str, port: int, timeout: int = 120) -> None:
    """Poll until the port accepts connections (replaces unreliable sleep)."""
    deadline = time.time() + timeout
    attempt  = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, port), timeout=3):
                elapsed = time.time() - (deadline - timeout)
                logger.info(f"  Ray head at {host}:{port} ready (attempt {attempt}, {elapsed:.1f}s)")
                return
        except OSError:
            logger.info(f"  Waiting for Ray head {host}:{port} (attempt {attempt})...")
            time.sleep(3)
    raise TimeoutError(
        f"Ray head {host}:{port} not ready within {timeout}s. "
        "Check logs/ray_head_*.err for details."
    )


# ── SLURM bootstrap ────────────────────────────────────────────────────────────

def _bootstrap_ray_on_slurm(ray_port: int) -> None:
    """
    Start Ray head on this node, workers on remaining SLURM nodes.
    Only called when SLURM_NNODES > 1 and no existing Ray cluster.
    """
    head_node     = socket.gethostname()
    num_nodes     = int(os.environ.get("SLURM_NNODES", 1))
    cpus_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))
    gpus_per_node = torch.cuda.device_count()
    redis_pass    = os.environ.get("RAY_REDIS_PASSWORD", "")
    job_id        = os.environ.get("SLURM_JOB_ID", "local")
    dash_port     = int(os.environ.get("RAY_DASHBOARD_PORT", 8265))

    os.makedirs("logs", exist_ok=True)

    # 1. Start Ray head (background)
    head_cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={head_node}",
        f"--port={ray_port}",
        f"--num-gpus={gpus_per_node}",
        f"--num-cpus={cpus_per_node}",
        "--dashboard-host=0.0.0.0",
        f"--dashboard-port={dash_port}",
        "--disable-usage-stats",
        "--block",
    ]
    if redis_pass:
        head_cmd += ["--redis-password", redis_pass]

    logger.info(f"Starting Ray head on {head_node}:{ray_port} ({gpus_per_node} GPUs)")
    subprocess.Popen(
        head_cmd,
        stdout=open(f"logs/ray_head_{job_id}.out", "w"),
        stderr=open(f"logs/ray_head_{job_id}.err", "w"),
    )

    # 2. Wait until head's GCS port is accepting connections
    _wait_for_tcp(head_node, ray_port, timeout=120)

    # 3. Start workers on remaining nodes via srun --relative
    if num_nodes > 1:
        worker_cmd = (
            f"ray start "
            f"--address={head_node}:{ray_port} "
            f"--num-gpus={gpus_per_node} "
            f"--num-cpus={cpus_per_node} "
            "--block"
        )
        if redis_pass:
            worker_cmd += f" --redis-password {redis_pass}"

        srun_cmd = [
            "srun", "--relative=1",
            f"--ntasks={num_nodes - 1}",
            "--ntasks-per-node=1",
            "bash", "-c", worker_cmd,
        ]
        logger.info(f"Starting Ray workers on {num_nodes - 1} additional node(s)")
        subprocess.Popen(
            srun_cmd,
            stdout=open(f"logs/ray_workers_{job_id}.out", "w"),
            stderr=open(f"logs/ray_workers_{job_id}.err", "w"),
        )
        # Brief wait for workers to register with head
        logger.info("Waiting 15s for workers to register...")
        time.sleep(15)

    logger.info(
        f"Ray cluster ready | head={head_node}:{ray_port} "
        f"| dashboard=http://{head_node}:{dash_port} "
        f"(SSH tunnel: ssh -L {dash_port}:{head_node}:{dash_port} <login_node>)"
    )


# ── Ray init (environment-aware) ───────────────────────────────────────────────

def _init_ray(num_workers: int) -> None:
    """
    Connect to / start Ray, adapting to the runtime environment.

    Priority:
      1. RAY_ADDRESS env var  → connect to existing cluster (any environment)
      2. Multi-node SLURM     → bootstrap head+workers, connect via "auto"
      3. Everything else      → plain local ray.init() (works anywhere)
    """
    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized, reusing.")
        return

    ray_port = int(os.environ.get(
        "RAY_PORT",
        6379 + int(os.environ.get("SLURM_JOB_ID", 0)) % 1000,
    ))

    existing   = os.environ.get("RAY_ADDRESS")
    in_slurm   = "SLURM_JOB_ID" in os.environ
    num_nodes  = int(os.environ.get("SLURM_NNODES", 1))

    if existing:
        logger.info(f"Connecting to existing Ray cluster at {existing}")
        ray.init(address=existing, ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    elif in_slurm and num_nodes > 1:
        _bootstrap_ray_on_slurm(ray_port)
        logger.info("Connecting to Ray cluster via auto-discovery")
        ray.init(address="auto", ignore_reinit_error=True,
                 logging_level=logging.WARNING)

    else:
        # Works for: single node with multiple GPUs, laptop, cloud VM, Kubernetes pod
        logger.info(f"Starting local Ray instance ({num_workers} GPU workers requested)")
        ray.init(
            num_gpus=torch.cuda.device_count(),
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )

    resources    = ray.available_resources()
    visible_gpus = int(resources.get("GPU", 0))
    logger.info(f"Ray online | visible GPUs={visible_gpus} requested={num_workers}")
    if visible_gpus < num_workers:
        logger.warning(
            f"Only {visible_gpus}/{num_workers} GPUs visible to Ray. "
            "Training will block until resources are available."
        )


# ── Ray Train worker function ──────────────────────────────────────────────────

def _ray_train_func(train_loop_config: dict) -> None:
    """
    Runs inside each Ray actor (one per GPU).
    Ray Train has already called dist.init_process_group() before this runs.
    """
    import ray.train

    ctx_ray     = ray.train.get_context()
    local_rank  = ctx_ray.get_local_rank()
    global_rank = ctx_ray.get_world_rank()
    world_size  = ctx_ray.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Change to submit directory so relative config paths resolve correctly
    submit_dir = train_loop_config.get("submit_dir", ".")
    if os.path.isdir(submit_dir):
        os.chdir(submit_dir)

    from bhaskera.config_loader import load_config
    cfg = load_config(train_loop_config["config_path"])

    from bhaskera.launcher.worker_core import WorkerContext, run_worker
    ctx = WorkerContext(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        launcher="ray",
    )
    run_worker(ctx, cfg)


# ── main launcher ──────────────────────────────────────────────────────────────

def _launch(args: argparse.Namespace) -> None:
    import ray
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    num_workers = args.num_workers

    _init_ray(num_workers)

    run_name = args.run_name or f"bhaskera_{os.environ.get('SLURM_JOB_ID', 'local')}"

    trainer = TorchTrainer(
        train_loop_per_worker=_ray_train_func,
        train_loop_config={
            "config_path": args.config,
            "submit_dir":  os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1, "CPU": 2},
        ),
        torch_config=TorchConfig(backend="nccl", timeout_s=1800),
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.abspath(args.ray_results_dir),
            failure_config=FailureConfig(max_failures=args.max_failures),
            checkpoint_config=CheckpointConfig(num_to_keep=3),
        ),
    )

    logger.info(f"Launching TorchTrainer | workers={num_workers} run={run_name}")
    result = trainer.fit()

    if result.error:
        logger.error(f"Training failed: {result.error}")
        raise RuntimeError(str(result.error))

    logger.info(f"Training complete! Results saved to: {result.path}")
    ray.shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description="Bhaskera Ray Train entrypoint")
    p.add_argument("--config",          required=True,               help="YAML config path")
    p.add_argument("--num-workers",     type=int, default=1,         help="Number of GPU workers")
    p.add_argument("--max-failures",    type=int, default=2,         help="Max worker restart attempts")
    p.add_argument("--ray-results-dir", type=str, default="./ray_results")
    p.add_argument("--run-name",        type=str, default=None)
    _launch(p.parse_args())


if __name__ == "__main__":
    main()
