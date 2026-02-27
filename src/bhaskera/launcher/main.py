"""
bhaskera.launcher.main
=======================
Unified CLI entrypoint.  Selects the backend via --launcher flag,
which overrides the launcher field in the YAML config if present.

Usage:
    # SLURM backend (srun handles process spawning):
    srun python -m bhaskera.launcher.main --config config.yaml --launcher slurm

    # Ray backend (this process bootstraps Ray then launches TorchTrainer):
    python -m bhaskera.launcher.main --config config.yaml --launcher ray --num-workers 4

    # Use whatever is set in config YAML (launcher: slurm | ray):
    python -m bhaskera.launcher.main --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][main] %(levelname)s %(message)s")


def _resolve_launcher(args: argparse.Namespace) -> str:
    """
    Priority: --launcher flag > config YAML launcher field > default 'slurm'
    """
    if args.launcher:
        return args.launcher.lower()

    # Try reading from YAML without fully loading config
    try:
        import yaml
        with open(args.config) as f:
            data = yaml.safe_load(f)
        launcher = data.get("launcher", {})
        if isinstance(launcher, str):
            return launcher.lower()
        if isinstance(launcher, dict):
            return launcher.get("backend", "slurm").lower()
    except Exception:
        pass

    return "slurm"   # safe default — always works in SLURM environments


def main() -> None:
    p = argparse.ArgumentParser(
        description="Bhaskera unified training launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SLURM backend (called by srun, one process per GPU):
  srun python -m bhaskera.launcher.main --config config.yaml --launcher slurm

  # Ray backend (called once on head node, spawns workers):
  python -m bhaskera.launcher.main --config config.yaml --launcher ray --num-workers 4

  # Backend from YAML config field 'launcher: ray':
  python -m bhaskera.launcher.main --config config_ray.yaml
        """,
    )
    p.add_argument("--config",       required=True,              help="YAML config path")
    p.add_argument("--launcher",     choices=["slurm", "ray"],   help="Backend (overrides YAML)")
    p.add_argument("--num-workers",  type=int, default=None,     help="[Ray only] number of GPU workers")
    p.add_argument("--max-failures", type=int, default=2,        help="[Ray only] max worker restarts")
    p.add_argument("--ray-results-dir", type=str, default="./ray_results")
    p.add_argument("--run-name",     type=str, default=None)

    args = p.parse_args()
    launcher = _resolve_launcher(args)

    logger.info(f"Launcher backend: {launcher}")

    if launcher == "slurm":
        # SLURM path: this process IS the worker (srun puts it on a GPU)
        from bhaskera.launcher.slurm_entry import _run_worker
        _run_worker(args)

    elif launcher == "ray":
        # Ray path: this process is the driver — it bootstraps Ray and
        # submits the TorchTrainer job
        if args.num_workers is None:
            import torch
            args.num_workers = torch.cuda.device_count() or 1
            logger.info(f"--num-workers not set, defaulting to {args.num_workers}")

        from bhaskera.launcher.ray_entry import _launch
        _launch(args)

    else:
        logger.error(f"Unknown launcher: {launcher}")
        sys.exit(1)


if __name__ == "__main__":
    main()