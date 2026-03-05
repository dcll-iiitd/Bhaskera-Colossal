import wandb


class WandbLogger:
    def __init__(self, cfg):
        wandb.init(
            project=cfg.PROJECT,
            name=cfg.RUN_NAME,
            config=cfg.as_dict() if hasattr(cfg, "as_dict") else {},
        )

    def log(self, metrics, step: int) -> None:
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        wandb.finish()
