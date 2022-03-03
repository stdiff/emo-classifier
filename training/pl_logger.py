from typing import Dict, Any, Optional

import pandas as pd
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class SimpleLogger(LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.metrics = []
        self.params: Optional[Dict[str, Any]] = None

    @property
    def name(self):
        return "simple_logger"

    def flush(self):
        self.metrics = []
        self.params = None

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step):
        ## metrics = {"metric_name": 0.123, "epoch": 9}
        ## step will not be cleared at the end of an epoch. It is just increasing
        # print(metrics)
        metrics["step"] = step
        self.metrics.append(metrics)

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

    def get_history(self) -> pd.DataFrame:
        rows = []
        for metric in self.metrics:
            metric_name = [key for key in metric.keys() if key not in ("epoch", "step")][0]
            row = {
                "epoch": metric["epoch"],
                "step": metric["step"],
                "metric": metric_name,
                "value": metric[metric_name],
            }
            rows.append(row)

        return pd.DataFrame(rows)
