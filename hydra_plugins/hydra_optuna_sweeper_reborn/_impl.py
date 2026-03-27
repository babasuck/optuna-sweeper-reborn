import functools
import logging
import sys
import warnings
from textwrap import dedent
from typing import Any, Callable, Dict, List, MutableSequence, Optional, Sequence, Tuple

import optuna
from hydra._internal.deprecation_warning import deprecation_warning
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, TaskFunction
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import Trial

from ._callbacks import BestTrialCallback, LogProgressCallback
from ._distributions import (
    create_optuna_distribution_from_config,
    create_params_from_overrides,
)
from .config import Direction

log = logging.getLogger(__name__)


class OptunaSweeperImpl(Sweeper):
    def __init__(
        self,
        sampler: Any,
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
        self.sampler = sampler
        self.direction = direction
        self.storage = storage
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.max_failure_rate = max_failure_rate
        assert 0.0 <= self.max_failure_rate <= 1.0
        self.custom_search_space_extender: Optional[
            Callable[[DictConfig, Trial], None]
        ] = None
        if custom_search_space:
            self.custom_search_space_extender = get_method(custom_search_space)
        self.search_space = search_space
        self.params = params
        self.job_idx: int = 0
        self.search_space_distributions: Optional[Dict[str, BaseDistribution]] = None

        # New fields
        self.pruner = pruner
        self.enable_pruning = enable_pruning
        self.dashboard_config = dashboard
        self.callbacks_config = callbacks

        self.task_function: Optional[TaskFunction] = None

    def _process_searchspace_config(self) -> None:
        url = "https://hydra.cc/docs/upgrades/1.1_to_1.2/changes_to_sweeper_config/"
        if self.params is None and self.search_space is None:
            self.params = OmegaConf.create({})
        elif self.search_space is not None:
            if self.params is not None:
                warnings.warn(
                    "Both hydra.sweeper.params and hydra.sweeper.search_space are configured."
                    "\nHydra will use hydra.sweeper.params for defining search space."
                    f"\n{url}"
                )
            else:
                deprecation_warning(
                    message=dedent(
                        f"""\
                        `hydra.sweeper.search_space` is deprecated and will be removed in a future release.
                        Please configure with `hydra.sweeper.params`.
                        {url}
                        """
                    ),
                )
                self.search_space_distributions = {
                    str(x): create_optuna_distribution_from_config(y)
                    for x, y in self.search_space.items()
                }

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.job_idx = 0
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, hydra_context=hydra_context, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def _get_directions(self) -> List[str]:
        if isinstance(self.direction, MutableSequence):
            return [d.name if isinstance(d, Direction) else d for d in self.direction]
        elif isinstance(self.direction, str):
            return [self.direction]
        return [self.direction.name]

    def _configure_trials(
        self,
        trials: List[Trial],
        search_space_distributions: Dict[str, BaseDistribution],
        fixed_params: Dict[str, Any],
    ) -> Sequence[Sequence[str]]:
        overrides = []
        for trial in trials:
            for param_name, distribution in search_space_distributions.items():
                assert type(param_name) is str
                trial._suggest(param_name, distribution)
            for param_name, value in fixed_params.items():
                trial.set_user_attr(param_name, value)

            if self.custom_search_space_extender:
                assert self.config is not None
                self.custom_search_space_extender(self.config, trial)

            overlap = trial.params.keys() & trial.user_attrs
            if len(overlap):
                raise ValueError(
                    "Overlapping fixed parameters and search space parameters found! "
                    f"Overlapping parameters: {list(overlap)}"
                )
            params = dict(trial.params)
            params.update(fixed_params)

            overrides.append(tuple(f"{name}={val}" for name, val in params.items()))
        return overrides

    def _parse_sweeper_params_config(self) -> List[str]:
        if not self.params:
            return []
        return [f"{k!s}={v}" for k, v in self.params.items()]

    def _to_grid_sampler_choices(self, distribution: BaseDistribution) -> Any:
        if isinstance(distribution, CategoricalDistribution):
            return distribution.choices
        elif isinstance(distribution, IntDistribution):
            step = distribution.step
            n_items = (distribution.high - distribution.low) // step
            return [distribution.low + i * step for i in range(n_items)]
        elif isinstance(distribution, FloatDistribution) and distribution.step:
            step = distribution.step
            n_items = int((distribution.high - distribution.low) // step)
            return [distribution.low + i * step for i in range(n_items)]
        else:
            raise ValueError("GridSampler only supports discrete distributions.")

    def _build_callbacks(self) -> List[Callable]:
        """Get or instantiate Optuna study callbacks from config."""
        if not self.callbacks_config:
            return []

        from hydra.utils import instantiate

        callbacks = []
        for cb_conf in self.callbacks_config:
            # Hydra may have already instantiated via _target_
            if callable(cb_conf) and not isinstance(cb_conf, (dict, DictConfig)):
                callbacks.append(cb_conf)
            else:
                callbacks.append(instantiate(cb_conf))
        return callbacks

    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Get or instantiate pruner from config."""
        if self.pruner is None:
            return None

        # Hydra may have already instantiated the pruner via _target_
        if isinstance(self.pruner, optuna.pruners.BasePruner):
            return self.pruner

        from hydra.utils import instantiate

        return instantiate(self.pruner)

    def _start_dashboard(self) -> Any:
        """Start Optuna Dashboard if configured."""
        if self.dashboard_config is None:
            return None

        enabled = (
            self.dashboard_config.get("enabled", False)
            if hasattr(self.dashboard_config, "get")
            else getattr(self.dashboard_config, "enabled", False)
        )
        if not enabled:
            return None

        if self.storage is None:
            log.warning(
                "Dashboard requires storage to be configured. Skipping dashboard launch."
            )
            return None

        from ._dashboard import DashboardManager

        host = (
            self.dashboard_config.get("host", "localhost")
            if hasattr(self.dashboard_config, "get")
            else getattr(self.dashboard_config, "host", "localhost")
        )
        port = (
            self.dashboard_config.get("port", 8080)
            if hasattr(self.dashboard_config, "get")
            else getattr(self.dashboard_config, "port", 8080)
        )
        manager = DashboardManager(
            storage=str(self.storage),
            host=host,
            port=port,
        )
        manager.start()
        return manager

    def _sweep_standard(
        self,
        study: optuna.Study,
        search_space_distributions: Dict[str, BaseDistribution],
        fixed_params: Dict[str, Any],
        directions: List[str],
        is_grid_sampler: bool,
        callbacks: List[Callable],
    ) -> None:
        """Standard sweep loop using ask/tell pattern (no pruning)."""
        batch_size = self.n_jobs
        n_trials_to_go = self.n_trials

        while n_trials_to_go > 0:
            batch_size = min(n_trials_to_go, batch_size)

            trials = [study.ask() for _ in range(batch_size)]
            overrides = self._configure_trials(
                trials, search_space_distributions, fixed_params
            )

            returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
            self.job_idx += len(returns)
            failures = []
            for trial, ret in zip(trials, returns):
                values: Optional[List[float]] = None
                state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
                try:
                    if len(directions) == 1:
                        try:
                            values = [float(ret.return_value)]
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Return value must be float-castable. Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                    else:
                        try:
                            values = [float(v) for v in ret.return_value]
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Return value must be a list or tuple of float-castable values."
                                f" Got '{ret.return_value}'."
                            ).with_traceback(sys.exc_info()[2])
                        if len(values) != len(directions):
                            raise ValueError(
                                "The number of the values and the number of the objectives are"
                                f" mismatched. Expect {len(directions)}, but actually {len(values)}."
                            )

                    try:
                        study.tell(trial=trial, state=state, values=values)
                    except RuntimeError as e:
                        if (
                            is_grid_sampler
                            and "`Study.stop` is supposed to be invoked" in str(e)
                        ):
                            pass
                        else:
                            raise e

                except Exception as e:
                    state = optuna.trial.TrialState.FAIL
                    study.tell(trial=trial, state=state, values=values)
                    log.warning(f"Failed experiment: {e}")
                    failures.append(e)

                # Invoke callbacks manually in standard mode
                frozen_trial = study.trials[-1]
                for cb in callbacks:
                    try:
                        cb(study, frozen_trial)
                    except Exception as cb_err:
                        log.warning(f"Callback error: {cb_err}")

            # Raise if too many failures
            if len(failures) / len(returns) > self.max_failure_rate:
                log.error(
                    f"Failed {len(failures)} times out of {len(returns)} "
                    f"with max_failure_rate={self.max_failure_rate}."
                )
                assert len(failures) > 0
                for ret in returns:
                    ret.return_value  # delegate raising to JobReturn

            n_trials_to_go -= batch_size

    def _sweep_with_pruning(
        self,
        study: optuna.Study,
        search_space_distributions: Dict[str, BaseDistribution],
        fixed_params: Dict[str, Any],
        directions: List[str],
        callbacks: List[Callable],
    ) -> None:
        """Sweep with pruning support using ask/tell pattern.

        Supports parallel execution with any launcher (including Ray) by
        injecting trial metadata via ``hydra.job.env_set`` environment
        variables. Remote workers reconstruct the Trial object from shared
        storage to call ``trial.report()`` and ``trial.should_prune()``.
        """
        batch_size = self.n_jobs
        n_trials_to_go = self.n_trials

        if self.storage is None:
            log.warning(
                "Pruning with n_jobs > 1 requires persistent storage. "
                "Falling back to n_jobs=1."
            )
            batch_size = 1

        while n_trials_to_go > 0:
            batch_size = min(n_trials_to_go, batch_size)

            trials = [study.ask() for _ in range(batch_size)]
            overrides = self._configure_trials(
                trials, search_space_distributions, fixed_params
            )

            # Inject trial metadata via env vars for remote/local workers
            enriched_overrides = []
            for trial, trial_overrides in zip(trials, overrides):
                env_overrides = list(trial_overrides) + [
                    f"+hydra.job.env_set.OPTUNA_TRIAL_ID={trial._trial_id}",
                    f"+hydra.job.env_set.OPTUNA_STUDY_NAME={study.study_name}",
                    f"+hydra.job.env_set.OPTUNA_STORAGE={self.storage or ''}",
                ]
                enriched_overrides.append(tuple(env_overrides))

            returns = self.launcher.launch(
                enriched_overrides, initial_job_idx=self.job_idx
            )
            self.job_idx += len(returns)

            failures = []
            for trial, ret in zip(trials, returns):
                values: Optional[List[float]] = None
                state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE
                try:
                    ret_value = ret.return_value

                    if ret_value is None:
                        state = optuna.trial.TrialState.PRUNED
                    elif len(directions) == 1:
                        try:
                            values = [float(ret_value)]
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"Return value must be float-castable. Got '{ret_value}'."
                            ).with_traceback(sys.exc_info()[2])
                    else:
                        try:
                            values = [float(v) for v in ret_value]
                        except (ValueError, TypeError):
                            raise ValueError(
                                "Return value must be a list or tuple of float-castable "
                                f"values. Got '{ret_value}'."
                            ).with_traceback(sys.exc_info()[2])

                    study.tell(trial=trial, state=state, values=values)

                except optuna.TrialPruned:
                    study.tell(
                        trial=trial,
                        state=optuna.trial.TrialState.PRUNED,
                    )
                    log.info(f"Trial {trial.number} was pruned.")
                except Exception as e:
                    state = optuna.trial.TrialState.FAIL
                    study.tell(trial=trial, state=state, values=values)
                    log.warning(f"Failed experiment: {e}")
                    failures.append(e)

                # Invoke callbacks
                frozen_trial = study.trials[-1]
                for cb in callbacks:
                    try:
                        cb(study, frozen_trial)
                    except Exception as cb_err:
                        log.warning(f"Callback error: {cb_err}")

            # Raise if too many failures
            if len(failures) / len(returns) > self.max_failure_rate:
                log.error(
                    f"Failed {len(failures)} times out of {len(returns)} "
                    f"with max_failure_rate={self.max_failure_rate}."
                )
                assert len(failures) > 0
                for ret in returns:
                    ret.return_value  # delegate raising to JobReturn

            n_trials_to_go -= batch_size

    def sweep(self, arguments: List[str]) -> None:
        assert self.config is not None
        assert self.launcher is not None
        assert self.hydra_context is not None

        self._process_searchspace_config()
        params_conf = self._parse_sweeper_params_config()
        params_conf.extend(arguments)

        is_grid_sampler = (
            isinstance(self.sampler, functools.partial)
            and self.sampler.func == optuna.samplers.GridSampler
        )

        (
            override_search_space_distributions,
            fixed_params,
        ) = create_params_from_overrides(params_conf)

        search_space_distributions: Dict[str, BaseDistribution] = {}
        if self.search_space_distributions:
            search_space_distributions = self.search_space_distributions.copy()
        search_space_distributions.update(override_search_space_distributions)

        if is_grid_sampler:
            search_space_for_grid_sampler = {
                name: self._to_grid_sampler_choices(distribution)
                for name, distribution in search_space_distributions.items()
            }
            self.sampler = self.sampler(search_space_for_grid_sampler)
            n_trial = 1
            for v in search_space_for_grid_sampler.values():
                n_trial *= len(v)
            self.n_trials = min(self.n_trials, n_trial)
            log.info(
                f"Updating num of trials to {self.n_trials} due to using GridSampler."
            )

        # Remove fixed parameters from Optuna search space
        for param_name in fixed_params:
            if param_name in search_space_distributions:
                del search_space_distributions[param_name]

        directions = self._get_directions()

        # Create pruner
        pruner = self._create_pruner()

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            pruner=pruner,
            directions=directions,
            load_if_exists=True,
        )
        log.info(f"Study name: {study.study_name}")
        log.info(f"Storage: {self.storage}")
        log.info(f"Sampler: {type(self.sampler).__name__}")
        log.info(f"Directions: {directions}")
        if pruner is not None:
            log.info(f"Pruner: {type(pruner).__name__}")
        if self.enable_pruning:
            log.info(
                f"Pruning mode: ENABLED (n_jobs={self.n_jobs}, "
                f"trial accessible via get_current_trial())"
            )

        # Build callbacks
        callbacks = self._build_callbacks()

        # Start dashboard
        dashboard_manager = self._start_dashboard()

        try:
            if self.enable_pruning:
                self._sweep_with_pruning(
                    study=study,
                    search_space_distributions=search_space_distributions,
                    fixed_params=fixed_params,
                    directions=directions,
                    callbacks=callbacks,
                )
            else:
                self._sweep_standard(
                    study=study,
                    search_space_distributions=search_space_distributions,
                    fixed_params=fixed_params,
                    directions=directions,
                    is_grid_sampler=is_grid_sampler,
                    callbacks=callbacks,
                )
        finally:
            if dashboard_manager is not None:
                dashboard_manager.stop()

        # Log and save results
        results_to_serialize: Dict[str, Any]
        if len(directions) < 2:
            try:
                best_trial = study.best_trial
                results_to_serialize = {
                    "name": "optuna",
                    "best_params": best_trial.params,
                    "best_value": best_trial.value,
                }
                log.info(f"Best parameters: {best_trial.params}")
                log.info(f"Best value: {best_trial.value}")
            except ValueError:
                results_to_serialize = {"name": "optuna", "best_params": {}, "best_value": None}
                log.warning("No completed trials found.")
        else:
            best_trials = study.best_trials
            pareto_front = [
                {"params": t.params, "values": t.values} for t in best_trials
            ]
            results_to_serialize = {
                "name": "optuna",
                "solutions": pareto_front,
            }
            log.info(f"Number of Pareto solutions: {len(best_trials)}")
            for t in best_trials:
                log.info(f"    Values: {t.values}, Params: {t.params}")

        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )
