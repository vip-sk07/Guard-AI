try:
    import mlflow
except ImportError:
    mlflow = None

import time

class SystemMonitor:
    def __init__(self, experiment_name: str = "GuardAI_Alignment"):
        self.experiment_name = experiment_name
        if mlflow is not None:
            try:
                mlflow.set_experiment(self.experiment_name)
            except Exception:
                pass

    def log_evaluation(self, prompt: str, latency_ms: float, is_refused: bool, tokens: int):
        if mlflow is not None:
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_metric("latency_ms", latency_ms)
                    mlflow.log_metric("tokens", tokens)
                    mlflow.log_param("is_refused", is_refused)
            except Exception:
                pass
