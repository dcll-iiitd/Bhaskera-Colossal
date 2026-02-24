"""
Logger factory — returns the appropriate Logger subclass based on cfg.TRACKER.
"""


def build_logger(cfg, log_gpu: bool = True, gpu_log_every_n_steps: int = 1):
    """
    Build and return a logger instance.

    Args:
        cfg:                    Bhaskera config object.
        log_gpu:                Forward to MLflowLogger — whether to log GPU stats.
        gpu_log_every_n_steps:  Forward to MLflowLogger — cadence for GPU polling.

    Returns:
        A logger with .log(metrics, step) and .finish() methods, or None.
    """
    tracker = (
        getattr(cfg, "TRACKER", None)
        or getattr(cfg, "tracker", None)
    )

    if tracker == "wandb":
        from .wandb_logger import WandbLogger
        return WandbLogger(cfg)

    elif tracker == "mlflow":
        from .mlflow_logger import MLflowLogger
        return MLflowLogger(
            cfg,
            log_gpu=log_gpu,
            gpu_log_every_n_steps=gpu_log_every_n_steps,
        )

    return None
