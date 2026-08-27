import pytest
from hydra.core.config_store import ConfigStore

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
    NSGAIIISamplerConfig,
    NSGAIISamplerConfig,
    OptunaSweeperConf,
    PatientPrunerConfig,
    PercentilePrunerConfig,
    QMCSamplerConfig,
    RandomSamplerConfig,
    SuccessiveHalvingPrunerConfig,
    ThresholdPrunerConfig,
    TPESamplerConfig,
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
        cfg = DistributionConfig(type=DistributionType.float, low=0.0, high=1.0, log=True)
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
            "tpe.yaml",
            "random.yaml",
            "cmaes.yaml",
            "nsgaii.yaml",
            "nsgaiii.yaml",
            "gp.yaml",
            "qmc.yaml",
            "grid.yaml",
            "bruteforce.yaml",
        ]
        for name in expected:
            assert name in items, f"{name} not registered"

    def test_pruners_registered(self):
        cs = ConfigStore.instance()
        items = cs.list("hydra/sweeper/pruner")
        expected = [
            "median.yaml",
            "hyperband.yaml",
            "percentile.yaml",
            "threshold.yaml",
            "patient.yaml",
            "successive_halving.yaml",
            "nop.yaml",
        ]
        for name in expected:
            assert name in items, f"{name} not registered"


class TestGroupComposition:
    """The registered sampler/pruner groups must actually be selectable."""

    @staticmethod
    def _compose(tmp_path, overrides):
        from hydra import compose, initialize_config_dir

        (tmp_path / "config.yaml").write_text(
            "defaults:\n"
            "  - override /hydra/sweeper: optuna_reborn\n"
            "x: 0.0\n"
            "hydra:\n"
            "  mode: MULTIRUN\n"
            "  sweeper:\n"
            "    params:\n"
            "      x: interval(-1.0, 1.0)\n"
        )
        with initialize_config_dir(config_dir=str(tmp_path), version_base="1.3"):
            return compose(config_name="config", return_hydra_config=True, overrides=overrides)

    def test_pruner_group_override_composes(self, tmp_path):
        """`override /hydra/sweeper/pruner: median` used to fail with
        'No match in the defaults list' - the group was registered but unreachable."""
        cfg = self._compose(tmp_path, ["hydra/sweeper/pruner=median"])
        assert cfg.hydra.sweeper.pruner._target_ == "optuna.pruners.MedianPruner"

    def test_pruner_defaults_to_none(self, tmp_path):
        cfg = self._compose(tmp_path, [])
        assert cfg.hydra.sweeper.pruner is None

    def test_sampler_group_still_composes(self, tmp_path):
        cfg = self._compose(tmp_path, ["hydra/sweeper/sampler=nsgaii"])
        assert cfg.hydra.sweeper.sampler._target_ == "optuna.samplers.NSGAIISampler"

    def test_grid_sampler_instantiates(self):
        """GridSampler takes `search_space` positionally, so its config must be
        partial - otherwise instantiation raises TypeError."""
        import functools

        import optuna
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        sampler = instantiate(OmegaConf.structured(GridSamplerConfig()))
        assert isinstance(sampler, functools.partial)
        assert isinstance(sampler({"x": [1, 2, 3]}), optuna.samplers.GridSampler)


class TestNoDeprecatedSamplerArguments:
    """Our sampler configs must not force arguments Optuna has deprecated.

    Fields listed in a structured config are passed on every instantiation, even
    when the user never touched them — which produced a FutureWarning per run on
    Optuna 4.9 and would become a TypeError in 6.0.
    """

    SAMPLERS = [
        TPESamplerConfig,
        RandomSamplerConfig,
        CmaEsSamplerConfig,
        NSGAIISamplerConfig,
        NSGAIIISamplerConfig,
        GPSamplerConfig,
        QMCSamplerConfig,
        BruteForceSamplerConfig,
    ]

    @pytest.mark.parametrize("cfg_cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_instantiation_emits_no_deprecation(self, cfg_cls):
        import warnings

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            instantiate(OmegaConf.structured(cfg_cls()))

        offenders = [
            str(w.message)
            for w in caught
            if issubclass(w.category, (DeprecationWarning, FutureWarning))
        ]
        assert not offenders, offenders

    @pytest.mark.parametrize("cfg_cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_every_field_exists_in_the_optuna_signature(self, cfg_cls):
        """Catches arguments Optuna renamed or dropped outright."""
        import inspect
        from dataclasses import fields

        import optuna

        target = cfg_cls()._target_.rsplit(".", 1)[-1]
        accepted = inspect.signature(getattr(optuna.samplers, target).__init__).parameters
        ours = {f.name for f in fields(cfg_cls)} - {"_target_", "_partial_"}

        assert ours <= set(accepted), sorted(ours - set(accepted))
