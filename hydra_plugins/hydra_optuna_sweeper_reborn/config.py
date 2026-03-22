from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    step: Optional[float] = None
    choices: Optional[List[Any]] = None


# ============================================================
# Sampler Configs
# ============================================================


@dataclass
class SamplerConfig:
    _target_: str = MISSING


@dataclass
class TPESamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.TPESampler"
    consider_prior: bool = True
    prior_weight: float = 1.0
    consider_magic_clip: bool = True
    consider_endpoints: bool = False
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    seed: Optional[int] = None
    multivariate: bool = False
    group: bool = False
    warn_independent_sampling: bool = True
    constant_liar: bool = False


@dataclass
class RandomSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.RandomSampler"
    seed: Optional[int] = None


@dataclass
class CmaEsSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.CmaEsSampler"
    sigma0: Optional[float] = None
    n_startup_trials: int = 1
    warn_independent_sampling: bool = True
    seed: Optional[int] = None
    consider_pruned_trials: bool = False
    restart_strategy: Optional[str] = None
    popsize: Optional[int] = None
    inc_popsize: int = -1
    use_separable_cma: bool = False
    with_margin: bool = False
    lr_adapt: bool = False


@dataclass
class NSGAIISamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.NSGAIISampler"
    population_size: int = 50
    mutation_prob: Optional[float] = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    seed: Optional[int] = None


@dataclass
class NSGAIIISamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.NSGAIIISampler"
    population_size: int = 50
    mutation_prob: Optional[float] = None
    crossover_prob: float = 0.9
    swapping_prob: float = 0.5
    seed: Optional[int] = None
    dividing_parameter: int = 3


@dataclass
class GPSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.GPSampler"
    seed: Optional[int] = None
    n_startup_trials: int = 10
    deterministic_objective: bool = False
    warn_independent_sampling: bool = True


@dataclass
class QMCSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.QMCSampler"
    qmc_type: str = "sobol"
    scramble: bool = False
    seed: Optional[int] = None
    warn_asynchronous_seeding: bool = True
    warn_independent_sampling: bool = True


@dataclass
class GridSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.GridSampler"
    seed: Optional[int] = None


@dataclass
class BruteForceSamplerConfig(SamplerConfig):
    _target_: str = "optuna.samplers.BruteForceSampler"
    seed: Optional[int] = None
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
    lower: Optional[float] = None
    upper: Optional[float] = None
    n_warmup_steps: int = 0
    interval_steps: int = 1


@dataclass
class PatientPrunerConfig(PrunerConfig):
    _target_: str = "optuna.pruners.PatientPruner"
    wrapped_pruner: Optional[Any] = None
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
    _target_: str = (
        "hydra_plugins.hydra_optuna_sweeper_reborn.optuna_sweeper.OptunaSweeper"
    )
    defaults: List[Any] = field(
        default_factory=lambda: [{"sampler": "tpe"}]
    )

    sampler: Any = MISSING
    pruner: Optional[Any] = None
    direction: Any = Direction.minimize
    storage: Optional[str] = None
    study_name: Optional[str] = None
    n_trials: int = 20
    n_jobs: int = 2
    max_failure_rate: float = 0.0

    # Parameter space
    params: Optional[Dict[str, str]] = None
    search_space: Optional[Dict[str, Any]] = None  # deprecated, backward compat
    custom_search_space: Optional[str] = None

    # New features
    enable_pruning: bool = False
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    callbacks: Optional[List[Any]] = None


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
