from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class Direction(Enum):
    minimize = 1
    maximize = 2


class DistributionType(Enum):
    int = 1
    float = 2
    categorical = 3


# ============================================================
# Distribution Config (backward compatible with original plugin)
# ============================================================


@dataclass
class DistributionConfig:
    type: DistributionType = MISSING
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list[Any] | None = None


# ============================================================
# Sampler Configs
# ============================================================


@dataclass
class SamplerConfig:
    _target_: str = MISSING


@dataclass
class TPESamplerConfig(SamplerConfig):
    # consider_prior, prior_weight, consider_magic_clip, consider_endpoints and
    # warn_independent_sampling are deprecated in Optuna 4.9 (removed in 6.0), so
    # they are deliberately absent: listing them here would force them into every
    # TPESampler this plugin builds and emit a FutureWarning per run.
    _target_: str = "optuna.samplers.TPESampler"
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    seed: int | None = None
    multivariate: bool = False
    group: bool = False
    constant_liar: bool = False


@dataclass
class RandomSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.RandomSampler"
    seed: int | None = None


@dataclass
class CmaEsSamplerConfig(SamplerConfig):
    # sigma0 and x0 are deprecated in Optuna 4.9 (removed in 6.0) — see the note
    # on TPESamplerConfig.
    _target_: str = "optuna.samplers.CmaEsSampler"
    n_startup_trials: int = 1
    warn_independent_sampling: bool = True
    seed: int | None = None
    consider_pruned_trials: bool = False
    restart_strategy: str | None = None
    popsize: int | None = None
    inc_popsize: int = -1
    use_separable_cma: bool = False
    with_margin: bool = False
    lr_adapt: bool = False


@dataclass
class NSGAIISamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.NSGAIISampler"
    population_size: int = 50
    mutation_prob: float | None = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    seed: int | None = None


@dataclass
class NSGAIIISamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.NSGAIIISampler"
    population_size: int = 50
    mutation_prob: float | None = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    seed: int | None = None
    dividing_parameter: int = 3


@dataclass
class GPSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.GPSampler"
    seed: int | None = None
    n_startup_trials: int = 10
    deterministic_objective: bool = False
    warn_independent_sampling: bool = True


@dataclass
class QMCSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.QMCSampler"
    qmc_type: str = "sobol"
    scramble: bool = False
    seed: int | None = None
    warn_asynchronous_seeding: bool = True
    warn_independent_sampling: bool = True


@dataclass
class GridSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.GridSampler"
    # GridSampler takes `search_space` positionally; the sweeper builds it from
    # the params config and calls the partial in `sweep()`.
    _partial_: bool = True
    seed: int | None = None


@dataclass
class BruteForceSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.BruteForceSampler"
    seed: int | None = None
    avoid_premature_stop: bool = False


# ============================================================
# Pruner Configs
# ============================================================


@dataclass
class PrunerConfig:
    _target_: str = MISSING


@dataclass
class MedianPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.MedianPruner"
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    n_min_trials: int = 1


@dataclass
class HyperbandPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.HyperbandPruner"
    min_resource: int = 1
    max_resource: str = "auto"
    reduction_factor: int = 3
    bootstrap_count: int = 0


@dataclass
class PercentilePrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.PercentilePruner"
    percentile: float = 25.0
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1
    n_min_trials: int = 1


@dataclass
class ThresholdPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.ThresholdPruner"
    lower: float | None = None
    upper: float | None = None
    n_warmup_steps: int = 0
    interval_steps: int = 1


@dataclass
class PatientPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.PatientPruner"
    wrapped_pruner: Any | None = None
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class SuccessiveHalvingPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.SuccessiveHalvingPruner"
    min_resource: str = "auto"
    reduction_factor: int = 4
    min_early_stopping_rate: int = 0
    bootstrap_count: int = 0


@dataclass
class NopPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.NopPruner"


# ============================================================
# Dashboard Config
# ============================================================


@dataclass
class DashboardConfig:
    enabled: bool = False
    host: str = "localhost"
    port: int = 8080


# ============================================================
# Main Sweeper Config
# ============================================================


@dataclass
class OptunaSweeperConf:
    _target_: str = "hydra_plugins.hydra_optuna_sweeper_reborn.optuna_sweeper.OptunaSweeper"
    # `_self_` first so that the sampler/pruner groups override the fields below.
    # Without it Hydra applies `_self_` last and the `pruner: None` field silently
    # wipes out whatever `override /hydra/sweeper/pruner: <name>` selected.
    defaults: list[Any] = field(
        default_factory=lambda: ["_self_", {"sampler": "tpe"}, {"pruner": None}]
    )

    sampler: Any = MISSING
    pruner: Any | None = None
    direction: Any = Direction.minimize
    storage: str | None = None
    study_name: str | None = None
    n_trials: int = 20
    n_jobs: int = 2
    max_failure_rate: float = 0.0

    # Parameter space
    params: dict[str, str] | None = None
    search_space: dict[str, Any] | None = None  # deprecated, backward compat
    custom_search_space: str | None = None

    # New features
    enable_pruning: bool = False
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    callbacks: list[Any] | None = None

    # Warm start: parameter sets to try before sampling kicks in. Re-queuing is
    # skipped for points a resumed study already holds.
    enqueue: list[dict[str, Any]] | None = None
    # How many best trials to list in optimization_results.yaml (0 = only best).
    results_top_n: int = 5


# ============================================================
# ConfigStore Registration
# ============================================================


def _register_configs() -> None:
    cs = ConfigStore.instance()

    # Main sweeper
    cs.store(
        group="hydra/sweeper",
        name="optuna_reborn",
        node=OptunaSweeperConf,
        provider="optuna_sweeper_reborn",
    )

    # Samplers
    cs.store(
        group="hydra/sweeper/sampler",
        name="tpe",
        node=TPESamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="random",
        node=RandomSamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="cmaes",
        node=CmaEsSamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="nsgaii",
        node=NSGAIISamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="nsgaiii",
        node=NSGAIIISamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="gp",
        node=GPSamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="qmc",
        node=QMCSamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="grid",
        node=GridSamplerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/sampler",
        name="bruteforce",
        node=BruteForceSamplerConfig,
        provider="optuna_sweeper_reborn",
    )

    # Pruners
    cs.store(
        group="hydra/sweeper/pruner",
        name="median",
        node=MedianPrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="hyperband",
        node=HyperbandPrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="percentile",
        node=PercentilePrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="threshold",
        node=ThresholdPrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="patient",
        node=PatientPrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="successive_halving",
        node=SuccessiveHalvingPrunerConfig,
        provider="optuna_sweeper_reborn",
    )
    cs.store(
        group="hydra/sweeper/pruner",
        name="nop",
        node=NopPrunerConfig,
        provider="optuna_sweeper_reborn",
    )


_register_configs()
