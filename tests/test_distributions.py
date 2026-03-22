from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from hydra_plugins.hydra_optuna_sweeper_reborn._distributions import (
    create_optuna_distribution_from_config,
    create_params_from_overrides,
)


class TestCreateFromConfig:
    def test_float_distribution(self):
        config = {"type": "float", "low": 0.0, "high": 1.0}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, FloatDistribution)
        assert dist.low == 0.0
        assert dist.high == 1.0
        assert dist.log is False

    def test_float_log_distribution(self):
        config = {"type": "float", "low": 0.001, "high": 1.0, "log": True}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, FloatDistribution)
        assert dist.log is True

    def test_float_step_distribution(self):
        config = {"type": "float", "low": 0.0, "high": 1.0, "step": 0.1}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, FloatDistribution)
        assert dist.step == 0.1

    def test_int_distribution(self):
        config = {"type": "int", "low": 1, "high": 10}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, IntDistribution)
        assert dist.low == 1
        assert dist.high == 10

    def test_int_log_distribution(self):
        config = {"type": "int", "low": 1, "high": 100, "log": True}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, IntDistribution)
        assert dist.log is True

    def test_categorical_distribution(self):
        config = {"type": "categorical", "choices": ["a", "b", "c"]}
        dist = create_optuna_distribution_from_config(config)
        assert isinstance(dist, CategoricalDistribution)
        assert list(dist.choices) == ["a", "b", "c"]


class TestCreateFromOverrides:
    def test_interval_float(self):
        dists, fixed = create_params_from_overrides(["x=interval(0.0, 1.0)"])
        assert "x" in dists
        assert isinstance(dists["x"], FloatDistribution)
        assert dists["x"].low == 0.0
        assert dists["x"].high == 1.0

    def test_interval_int(self):
        dists, fixed = create_params_from_overrides(["x=interval(1, 10)"])
        assert "x" in dists
        # Hydra parses unquoted integers in interval() as int,
        # which creates IntDistribution
        assert isinstance(dists["x"], (IntDistribution, FloatDistribution))

    def test_interval_log(self):
        dists, fixed = create_params_from_overrides(
            ["x=tag(log, interval(0.001, 1.0))"]
        )
        assert "x" in dists
        assert isinstance(dists["x"], FloatDistribution)
        assert dists["x"].log is True

    def test_choice(self):
        dists, fixed = create_params_from_overrides(["x=choice(a, b, c)"])
        assert "x" in dists
        assert isinstance(dists["x"], CategoricalDistribution)

    def test_range(self):
        dists, fixed = create_params_from_overrides(["x=range(0, 10, 2)"])
        assert "x" in dists
        assert isinstance(dists["x"], IntDistribution)

    def test_fixed_param(self):
        dists, fixed = create_params_from_overrides(["x=5"])
        assert "x" in fixed
        assert len(dists) == 0

    def test_mixed(self):
        dists, fixed = create_params_from_overrides(
            ["x=interval(0.0, 1.0)", "y=42", "z=choice(a, b)"]
        )
        assert "x" in dists
        assert "z" in dists
        assert "y" in fixed
