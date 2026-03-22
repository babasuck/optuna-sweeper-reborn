from typing import Any, List, Optional

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
        storage: Optional[str],
        study_name: Optional[str],
        n_trials: int,
        n_jobs: int,
        max_failure_rate: float,
        search_space: Optional[DictConfig],
        custom_search_space: Optional[str],
        params: Optional[DictConfig],
        # New parameters
        pruner: Optional[Any] = None,
        enable_pruning: bool = False,
        dashboard: Optional[DictConfig] = None,
        callbacks: Optional[List[Any]] = None,
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
        )

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.sweeper.setup(
            hydra_context=hydra_context, task_function=task_function, config=config
        )

    def sweep(self, arguments: List[str]) -> None:
        return self.sweeper.sweep(arguments)
