import logging

logger = logging.getLogger(__name__)


def build_logger(cfg, log_gpu: bool = True, gpu_log_every_n_steps: int = 10):
    """Build experiment logger based on cfg.TRACKER."""
    tracker = (cfg.TRACKER or "").lower()

    if tracker == "mlflow":
        from .mlflow_logger import MLflowLogger
        return MLflowLogger(cfg, log_gpu=log_gpu, gpu_log_every_n_steps=gpu_log_every_n_steps)
    elif tracker == "wandb":
        from .wandb_logger import WandbLogger
        return WandbLogger(cfg)
    else:
        logger.warning(f"Unknown or unset tracker '{cfg.TRACKER}', logging disabled.")
        return None
