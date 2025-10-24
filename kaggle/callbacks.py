import os
from typing import Dict

import numpy as np
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


@PublicAPI
class ModelLoggerCallback(LoggerCallback):
    """
    A custom TensorBoardX Logger that logs metrics for a specific model.
    It creates a `SummaryWriter` in the trial's log directory.
    """

    def __init__(self):
        if SummaryWriter is None:
            raise ImportError('Run `pip install tensorboardx` to use TensorBoard callbacks.')
        self._trial_writer: Dict[Trial, SummaryWriter] = {}

    def log_trial_start(self, trial: Trial):
        # Close any existing writer
        if trial in self._trial_writer:
            self._trial_writer[trial].close()

        # Create a new writer in the trial's directory
        trial.init_local_path()
        self._trial_writer[trial] = SummaryWriter(trial.local_path, flush_secs=30)

    def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
        if trial not in self._trial_writer:
            self.log_trial_start(trial)

        writer = self._trial_writer[trial]
        step = result.get("timesteps_total") or result["training_iteration"]

        # Flatten the result dict to handle nested metrics
        flat_result = flatten_dict(result, delimiter="/")

        # Log scalar values
        for attr, value in flat_result.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"tune/{attr}", value, global_step=step)
        
        # Log histograms
        if "histograms" in result:
            for name, hist in result["histograms"].items():
                if isinstance(hist, np.ndarray):
                    writer.add_histogram(name, hist, global_step=step)

        writer.flush()

    def log_trial_end(self, trial: Trial, failed: bool = False):
        if trial in self._trial_writer:
            self._trial_writer[trial].close()
            del self._trial_writer[trial]
