import mlflow

class MLflowLogger:
    def __init__(self, cfg):
        mlflow.start_run(run_name=cfg.run_name)
        for k, v in cfg.as_dict().items():
            mlflow.log_param(k, v)

    def log(self, metrics, step):
        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=step)

    def finish(self):
        mlflow.end_run()
