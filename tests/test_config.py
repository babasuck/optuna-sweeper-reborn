from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from hydra_plugins.hydra_optuna_sweeper_reborn.config import (
    BruteForceSamplerConfig,
    CmaEsSamplerConfig,
    DashboardConfig,
    Direction,
    DistributionConfig,
    DistributionType,
    GPSamplerConfig,
    GridSamplerConfig,
    HyperbandPrunerConfig,
    MedianPrunerConfig,
    NopPrunerConfig,
    NSGAIISamplerConfig,
    NSGAIIISamplerConfig,
    OptunaSweeperConf,
    PatientPrunerConfig,
    PercentilePrunerConfig,
    QMCSamplerConfig,
    RandomSamplerConfig,
    SuccessiveHalvingPrunerConfig,
    TPESamplerConfig,
    ThresholdPrunerConfig,
)


class TestSamplerConfigs:
    def test_tpe_sampler_defaults(self):
        cfg = TPESamplerConfig()
        assert cfg._target_ == "optuna.samplers.TPESampler"
        assert cfg.n_startup_trials == 10
        assert cfg.seed is None
        assert cfg.multivariate is False

    def test_random_sampler_defaults(self):
        cfg = RandomSamplerConfig()
        assert cfg._target_ == "optuna.samplers.RandomSampler"
        assert cfg.seed is None

    def test_cmaes_sampler_defaults(self):
        cfg = CmaEsSamplerConfig()
        assert cfg._target_ == "optuna.samplers.CmaEsSampler"
        assert cfg.n_startup_trials == 1

    def test_nsgaii_sampler_defaults(self):
        cfg = NSGAIISamplerConfig()
        assert cfg._target_ == "optuna.samplers.NSGAIISampler"
        assert cfg.population_size == 50

    def test_nsgaiii_sampler_defaults(self):
        cfg = NSGAIIISamplerConfig()
        assert cfg._target_ == "optuna.samplers.NSGAIIISampler"
        assert cfg.dividing_parameter == 3

    def test_gp_sampler_defaults(self):
        cfg = GPSamplerConfig()
        assert cfg._target_ == "optuna.samplers.GPSampler"
        assert cfg.n_startup_trials == 10

    def test_qmc_sampler_defaults(self):
        cfg = QMCSamplerConfig()
        assert cfg._target_ == "optuna.samplers.QMCSampler"
        assert cfg.qmc_type == "sobol"

    def test_grid_sampler_defaults(self):
        cfg = GridSamplerConfig()
        assert cfg._target_ == "optuna.samplers.GridSampler"

    def test_bruteforce_sampler_defaults(self):
        cfg = BruteForceSamplerConfig()
        assert cfg._target_ == "optuna.samplers.BruteForceSampler"


class TestPrunerConfigs:
    def test_median_pruner_defaults(self):
        cfg = MedianPrunerConfig()
        assert cfg._target_ == "optuna.pruners.MedianPruner"
        assert cfg.n_startup_trials == 5
        assert cfg.n_warmup_steps == 0

    def test_hyperband_pruner_defaults(self):
        cfg = HyperbandPrunerConfig()
        assert cfg._target_ == "optuna.pruners.HyperbandPruner"
        assert cfg.reduction_factor == 3

    def test_percentile_pruner_defaults(self):
        cfg = PercentilePrunerConfig()
        assert cfg.percentile == 25.0

    def test_threshold_pruner_defaults(self):
        cfg = ThresholdPrunerConfig()
        assert cfg.lower is None
        assert cfg.upper is None

    def test_patient_pruner_defaults(self):
        cfg = PatientPrunerConfig()
        assert cfg.patience == 10

    def test_successive_halving_pruner_defaults(self):
        cfg = SuccessiveHalvingPrunerConfig()
        assert cfg.reduction_factor == 4

    def test_nop_pruner_defaults(self):
        cfg = NopPrunerConfig()
        assert cfg._target_ == "optuna.pruners.NopPruner"


class TestMainConfig:
    def test_sweeper_conf_defaults(self):
        cfg = OptunaSweeperConf()
        assert cfg.n_trials == 20
        assert cfg.n_jobs == 2
        assert cfg.max_failure_rate == 0.0
        assert cfg.enable_pruning is False
        assert cfg.pruner is None
        assert cfg.storage is None
        assert cfg.callbacks is None

    def test_direction_enum(self):
        assert Direction.minimize.name == "minimize"
        assert Direction.maximize.name == "maximize"

    def test_distribution_config(self):
        cfg = DistributionConfig(
            type=DistributionType.float, low=0.0, high=1.0, log=True
        )
        assert cfg.type == DistributionType.float
        assert cfg.log is True

    def test_dashboard_config_defaults(self):
        cfg = DashboardConfig()
        assert cfg.enabled is False
        assert cfg.host == "localhost"
        assert cfg.port == 8080


class TestConfigStore:
    def test_sweeper_registered(self):
        cs = ConfigStore.instance()
        items = cs.list("hydra/sweeper")
        assert "optuna_reborn.yaml" in items

    def test_samplers_registered(self):
        cs = ConfigStore.instance()
        items = cs.list("hydra/sweeper/sampler")
        expected = [
            "tpe.yaml", "random.yaml", "cmaes.yaml",
            "nsgaii.yaml", "nsgaiii.yaml", "gp.yaml",
            "qmc.yaml", "grid.yaml", "bruteforce.yaml",
        ]
        for name in expected:
            assert name in items, f"{name} not registered"

    def test_pruners_registered(self):
        cs = ConfigStore.instance()
        items = cs.list("hydra/sweeper/pruner")
        expected = [
            "median.yaml", "hyperband.yaml", "percentile.yaml",
            "threshold.yaml", "patient.yaml", "successive_halving.yaml",
            "nop.yaml",
        ]
        for name in expected:
            assert name in items, f"{name} not registered"
