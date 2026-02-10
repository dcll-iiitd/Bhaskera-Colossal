def build_logger(cfg):
    if cfg.tracker == "wandb":
        from .wandb_logger import WandbLogger
        return WandbLogger(cfg)
    elif cfg.tracker == "mlflow":
        from .mlflow_logger import MLflowLogger
        return MLflowLogger(cfg)
    else:
        return None
