import wandb

class WandbLogger:
    def __init__(self, cfg):
        wandb.init(
            project=cfg.project,
            name=cfg.run_name,
            config=cfg.as_dict(),
        )

    def log(self, metrics, step):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()

