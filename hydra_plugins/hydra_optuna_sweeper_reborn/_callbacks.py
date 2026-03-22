import logging

from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

log = logging.getLogger(__name__)


class LogProgressCallback:
    """Logs progress after each trial completion."""

    def __init__(self, log_interval: int = 1) -> None:
        self.log_interval = log_interval

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        if trial.number % self.log_interval != 0:
            return
        n_complete = len(
            [t for t in study.trials if t.state == TrialState.COMPLETE]
        )
        n_pruned = len(
            [t for t in study.trials if t.state == TrialState.PRUNED]
        )
        n_fail = len(
            [t for t in study.trials if t.state == TrialState.FAIL]
        )
        best_str = "N/A"
        if n_complete > 0:
            try:
                best_str = str(study.best_value)
            except ValueError:
                best_str = "N/A (multi-objective)"
        log.info(
            f"Trial {trial.number} finished. "
            f"Complete: {n_complete}, Pruned: {n_pruned}, Failed: {n_fail}, "
            f"Best value: {best_str}"
        )


class BestTrialCallback:
    """Logs when a new best trial is found."""

    def __init__(self) -> None:
        self._best_value = None

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        try:
            current_best = study.best_value
            if self._best_value is None or current_best != self._best_value:
                self._best_value = current_best
                log.info(
                    f"New best trial! Number: {study.best_trial.number}, "
                    f"Value: {current_best}, Params: {study.best_trial.params}"
                )
        except ValueError:
            pass
