import pytest
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobReturn, JobStatus
from omegaconf import OmegaConf

from hydra_plugins.hydra_optuna_sweeper_reborn._impl import OptunaSweeperImpl


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra global state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def make_job_return(value):
    """Build a JobReturn as a launcher would.

    Pass an exception instance to model a job that raised (including
    ``optuna.TrialPruned``); anything else is a normal return value.
    """
    if isinstance(value, BaseException):
        return JobReturn(
            overrides=[], status=JobStatus.FAILED, _return_value=value
        )
    return JobReturn(overrides=[], status=JobStatus.COMPLETED, _return_value=value)


class FakeLauncher:
    """Launcher that replays a canned list of results per batch.

    Lets the sweep loop be exercised in-process instead of through the slow
    subprocess integration tests.
    """

    def __init__(self, results):
        self.results = list(results)
        self.launched = []

    def launch(self, job_overrides, initial_job_idx):
        job_overrides = list(job_overrides)
        self.launched.extend(job_overrides)
        batch, self.results = (
            self.results[: len(job_overrides)],
            self.results[len(job_overrides) :],
        )
        return [make_job_return(v) for v in batch]


@pytest.fixture
def make_sweeper():
    """Build an OptunaSweeperImpl wired to a FakeLauncher, bypassing Hydra setup."""

    def _make(results, *, n_trials, n_jobs=1, directions="minimize",
              max_failure_rate=0.0, enable_pruning=False, sweeper_cfg=None):
        sweeper = OptunaSweeperImpl(
            sampler=None,
            direction=directions,
            storage=None,
            study_name=None,
            n_trials=n_trials,
            n_jobs=n_jobs,
            max_failure_rate=max_failure_rate,
            search_space=None,
            custom_search_space=None,
            params=None,
            enable_pruning=enable_pruning,
        )
        sweeper.launcher = FakeLauncher(results)
        sweeper.config = OmegaConf.create(
            {"hydra": {"sweeper": sweeper_cfg or {}, "sweep": {"dir": "."}}}
        )
        return sweeper

    return _make
