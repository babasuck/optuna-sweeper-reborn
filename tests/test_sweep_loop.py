"""Regression tests for the sweep loop. Each one fails on the pre-fix code."""

import logging

import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from hydra_plugins.hydra_optuna_sweeper_reborn._impl import OptunaSweeperImpl


def _run(sweeper, study, directions, callbacks=None, dists=None):
    sweeper._sweep(
        study=study,
        search_space_distributions=dists or {},
        fixed_params={},
        directions=directions,
        is_grid_sampler=False,
        callbacks=callbacks or [],
    )


class TestGridChoices:
    """The grid must include the upper bound of the distribution."""

    @pytest.fixture
    def impl(self):
        return OptunaSweeperImpl.__new__(OptunaSweeperImpl)

    def test_int_choices_include_upper_bound(self, impl):
        assert impl._to_grid_sampler_choices(IntDistribution(1, 5, step=1)) == [
            1, 2, 3, 4, 5
        ]

    def test_int_choices_with_step(self, impl):
        assert impl._to_grid_sampler_choices(IntDistribution(0, 10, step=5)) == [
            0, 5, 10
        ]

    def test_float_choices_include_upper_bound(self, impl):
        choices = impl._to_grid_sampler_choices(
            FloatDistribution(0.0, 1.0, step=0.25)
        )
        assert choices == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


class TestFailureHandling:
    def test_value_count_mismatch_fails_trial_not_sweep(self, make_sweeper):
        """Returning one value for two directions must fail the trial, not blow
        up the whole sweep with 'Values cannot be specified when state is FAIL'."""
        sweeper = make_sweeper([[1.0]], n_trials=1, max_failure_rate=1.0)
        study = optuna.create_study(directions=["minimize", "minimize"])

        _run(sweeper, study, ["minimize", "minimize"])

        assert [t.state for t in study.trials] == [TrialState.FAIL]

    def test_non_castable_value_fails_trial(self, make_sweeper):
        sweeper = make_sweeper(["not-a-number"], n_trials=1, max_failure_rate=1.0)
        study = optuna.create_study()

        _run(sweeper, study, ["minimize"])

        assert [t.state for t in study.trials] == [TrialState.FAIL]

    def test_none_return_is_fail_not_pruned(self, make_sweeper):
        """A forgotten `return` is an error, not a silent prune."""
        sweeper = make_sweeper([None], n_trials=1, max_failure_rate=1.0)
        study = optuna.create_study()

        _run(sweeper, study, ["minimize"])

        assert [t.state for t in study.trials] == [TrialState.FAIL]

    def test_pruned_job_is_told_as_pruned(self, make_sweeper):
        sweeper = make_sweeper([optuna.TrialPruned()], n_trials=1)
        study = optuna.create_study()

        _run(sweeper, study, ["minimize"])

        assert [t.state for t in study.trials] == [TrialState.PRUNED]

    def test_max_failure_rate_still_raises(self, make_sweeper):
        sweeper = make_sweeper(
            [RuntimeError("boom")], n_trials=1, max_failure_rate=0.0
        )
        study = optuna.create_study()

        with pytest.raises(RuntimeError, match="boom"):
            _run(sweeper, study, ["minimize"])


class TestCallbacks:
    def test_callback_receives_the_trial_that_was_told(self, make_sweeper):
        """With n_jobs > 1, study.trials[-1] is the last *asked* trial (still
        RUNNING), not the one just told."""
        seen = []
        sweeper = make_sweeper([5.0, 1.0, 9.0], n_trials=3, n_jobs=3)
        study = optuna.create_study(direction="minimize")

        _run(sweeper, study, ["minimize"], callbacks=[
            lambda s, t: seen.append((t.number, t.state, t.value))
        ])

        assert [n for n, _, _ in seen] == [0, 1, 2]
        assert all(state == TrialState.COMPLETE for _, state, _ in seen)
        assert [v for _, _, v in seen] == [5.0, 1.0, 9.0]

    def test_callbacks_survive_multi_objective_study(self, make_sweeper, caplog):
        """The built-in callbacks must not blow up on a multi-objective study
        (study.best_value raises RuntimeError there, not ValueError)."""
        from hydra_plugins.hydra_optuna_sweeper_reborn._callbacks import (
            BestTrialCallback,
            LogProgressCallback,
        )

        sweeper = make_sweeper([[1.0, 2.0]], n_trials=1)
        study = optuna.create_study(directions=["minimize", "minimize"])

        with caplog.at_level(logging.WARNING):
            _run(
                sweeper,
                study,
                ["minimize", "minimize"],
                callbacks=[BestTrialCallback(), LogProgressCallback()],
            )

        assert "Callback error" not in caplog.text
        assert [t.state for t in study.trials] == [TrialState.COMPLETE]


class TestPruningValidation:
    def test_pruning_with_multi_objective_raises_early(self, make_sweeper):
        """Trial.report/should_prune raise NotImplementedError for multi-objective,
        so the sweeper must refuse up front instead of failing every trial."""
        sweeper = make_sweeper(
            [], n_trials=2, directions=["minimize", "minimize"], enable_pruning=True
        )
        sweeper.hydra_context = object()
        sweeper.params = {"x": "interval(0.0, 1.0)"}

        with pytest.raises(ValueError, match="enable_pruning is not supported"):
            sweeper.sweep([])


class TestBatchWarnings:
    def _sweeper_with_sampler(self, make_sweeper, sampler_cfg):
        return make_sweeper(
            [1.0, 2.0],
            n_trials=2,
            n_jobs=2,
            sweeper_cfg={"sampler": sampler_cfg},
        )

    def test_warns_when_tpe_batches_without_constant_liar(
        self, make_sweeper, caplog
    ):
        sweeper = self._sweeper_with_sampler(
            make_sweeper, {"_target_": "optuna.samplers.TPESampler"}
        )
        with caplog.at_level(logging.WARNING):
            sweeper._warn_about_batched_sampling()
        assert "constant_liar" in caplog.text

    def test_no_warning_when_constant_liar_is_set(self, make_sweeper, caplog):
        sweeper = self._sweeper_with_sampler(
            make_sweeper,
            {"_target_": "optuna.samplers.TPESampler", "constant_liar": True},
        )
        with caplog.at_level(logging.WARNING):
            sweeper._warn_about_batched_sampling()
        assert "constant_liar" not in caplog.text

    def test_warns_about_batch_barrier_when_pruning(self, make_sweeper, caplog):
        sweeper = make_sweeper(
            [],
            n_trials=2,
            n_jobs=2,
            enable_pruning=True,
            sweeper_cfg={"sampler": {"_target_": "optuna.samplers.RandomSampler"}},
        )
        with caplog.at_level(logging.WARNING):
            sweeper._warn_about_batched_sampling()
        assert "slowest job" in caplog.text


class TestPrunerPublication:
    def test_pruner_config_is_published_to_the_study(self, make_sweeper, tmp_path):
        import json

        from hydra_plugins.hydra_optuna_sweeper_reborn._trial_provider import (
            PRUNER_USER_ATTR,
        )

        sweeper = make_sweeper([], n_trials=1)
        sweeper.pruner_config = {
            "_target_": "optuna.pruners.MedianPruner",
            "n_startup_trials": 11,
        }
        study = optuna.create_study(
            study_name="pub", storage=f"sqlite:///{tmp_path}/p.db"
        )

        sweeper._publish_pruner_config(study)

        stored = json.loads(study.user_attrs[PRUNER_USER_ATTR])
        assert stored["n_startup_trials"] == 11

    def test_env_injection_carries_no_pruner_payload(self, make_sweeper):
        """The pruner rides in storage; env vars stay small and log-friendly."""
        sweeper = make_sweeper([], n_trials=1, enable_pruning=True)
        sweeper.storage = "sqlite:///x.db"
        study = optuna.create_study()
        trial = study.ask()

        overrides = sweeper._inject_trial_env([trial], [("a=1",)], study)

        joined = " ".join(overrides[0])
        assert "OPTUNA_PRUNER" not in joined
        assert "OPTUNA_TRIAL_ID" in joined
        assert len(joined) < 200
