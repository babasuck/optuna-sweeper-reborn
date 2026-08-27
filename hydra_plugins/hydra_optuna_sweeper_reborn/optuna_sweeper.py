from typing import Any

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from .config import SamplerConfig


class OptunaSweeper(Sweeper):
    """Hydra Sweeper plugin for Optuna with pruning, dashboard, and callbacks."""

    def __init__(
        self,
        sampler: SamplerConfig,
        direction: Any,
        storage: str | None,
        study_name: str | None,
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        search_space: DictConfig | None,
        custom_search_space: str | None,
        params: DictConfig | None,
        # New parameters
        pruner: Any | None = None,
        enable_pruning: bool = False,
        dashboard: DictConfig | None = None,
        callbacks: list[Any] | None = None,
        enqueue: list[Any] | None = None,
        results_top_n: int = 5,
    ) -> None:
        from ._impl import OptunaSweeperImpl

        self.sweeper = OptunaSweeperImpl(
            sampler=sampler,
            direction=direction,
            storage=storage,
            study_name=study_name,
            n_trials=n_trials,
            n_jobs=n_jobs,
            max_failure_rate=max_failure_rate,
            search_space=search_space,
            custom_search_space=custom_search_space,
            params=params,
            pruner=pruner,
            enable_pruning=enable_pruning,
            dashboard=dashboard,
            callbacks=callbacks,
            enqueue=enqueue,
            results_top_n=results_top_n,
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments: list[str]) -> None:
        return self.sweeper.sweep(arguments)
