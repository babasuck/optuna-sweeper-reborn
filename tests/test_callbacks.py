from unittest.mock import MagicMock

import optuna
from optuna.trial import FrozenTrial, TrialState

from hydra_plugins.hydra_optuna_sweeper_reborn._callbacks import (
    BestTrialCallback,
    LogProgressCallback,
)


def _make_study_and_trial(value=1.0, state=TrialState.COMPLETE):
    study = optuna.create_study()

    def objective(trial):
        return value

    study.optimize(objective, n_trials=1)
    trial = study.trials[-1]
    return study, trial


class TestLogProgressCallback:
    def test_logs_on_interval(self):
        cb = LogProgressCallback(log_interval=1)
        study, trial = _make_study_and_trial()
        # Should not raise
        cb(study, trial)

    def test_skips_between_intervals(self):
        cb = LogProgressCallback(log_interval=5)
        study, trial = _make_study_and_trial()
        # trial.number is 0, 0 % 5 == 0 so it should log
        cb(study, trial)


class TestBestTrialCallback:
    def test_detects_new_best(self):
        cb = BestTrialCallback()
        study, trial = _make_study_and_trial(value=5.0)
        # First call — new best
        cb(study, trial)
        assert cb._best_value == 5.0

    def test_updates_on_better(self):
        cb = BestTrialCallback()
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: 5.0, n_trials=1)
        cb(study, study.trials[-1])
        assert cb._best_value == 5.0

        study.optimize(lambda t: 3.0, n_trials=1)
        cb(study, study.trials[-1])
        assert cb._best_value == 3.0

    def test_ignores_pruned(self):
        cb = BestTrialCallback()
        study = optuna.create_study()
        trial = MagicMock(spec=FrozenTrial)
        trial.state = TrialState.PRUNED
        # Should not raise or update
        cb(study, trial)
        assert cb._best_value is None
