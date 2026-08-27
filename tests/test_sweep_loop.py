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
        assert impl._to_grid_sampler_choices(IntDistribution(1, 5, step=1)) == [1, 2, 3, 4, 5]

    def test_int_choices_with_step(self, impl):
        assert impl._to_grid_sampler_choices(IntDistribution(0, 10, step=5)) == [0, 5, 10]

    def test_float_choices_include_upper_bound(self, impl):
        choices = impl._to_grid_sampler_choices(FloatDistribution(0.0, 1.0, step=0.25))
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
        sweeper = make_sweeper([RuntimeError("boom")], n_trials=1, max_failure_rate=0.0)
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

        _run(
            sweeper,
            study,
            ["minimize"],
            callbacks=[lambda s, t: seen.append((t.number, t.state, t.value))],
        )

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

    def test_warns_when_tpe_batches_without_constant_liar(self, make_sweeper, caplog):
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
        study = optuna.create_study(study_name="pub", storage=f"sqlite:///{tmp_path}/p.db")

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


class TestWarmStart:
    def test_enqueued_params_are_used_first(self, make_sweeper):
        """Warm-start points must come out of the first ask()s, ahead of sampling."""
        dists = {"x": FloatDistribution(-5.0, 5.0)}
        sweeper = make_sweeper([1.0, 2.0, 3.0], n_trials=3, enqueue=[{"x": 4.25}, {"x": -3.5}])
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=0), direction="minimize"
        )

        sweeper._enqueue_initial_trials(study)
        _run(sweeper, study, ["minimize"], dists=dists)

        assert [t.params["x"] for t in study.trials[:2]] == [4.25, -3.5]

    def test_partial_enqueue_leaves_the_rest_sampled(self, make_sweeper):
        dists = {
            "x": FloatDistribution(-5.0, 5.0),
            "y": FloatDistribution(-5.0, 5.0),
        }
        sweeper = make_sweeper([1.0], n_trials=1, enqueue=[{"x": 2.0}])
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=0), direction="minimize"
        )

        sweeper._enqueue_initial_trials(study)
        _run(sweeper, study, ["minimize"], dists=dists)

        assert study.trials[0].params["x"] == 2.0
        assert "y" in study.trials[0].params

    def test_resume_does_not_requeue_the_same_point(self, make_sweeper):
        """Otherwise every restart of a long study burns trials on the same points."""
        dists = {"x": FloatDistribution(-5.0, 5.0)}
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=0), direction="minimize"
        )

        first = make_sweeper([1.0], n_trials=1, enqueue=[{"x": 4.25}])
        first._enqueue_initial_trials(study)
        _run(first, study, ["minimize"], dists=dists)

        second = make_sweeper([1.0], n_trials=1, enqueue=[{"x": 4.25}])
        second._enqueue_initial_trials(study)
        _run(second, study, ["minimize"], dists=dists)

        assert study.trials[1].params["x"] != 4.25

    def test_no_enqueue_is_a_noop(self, make_sweeper):
        sweeper = make_sweeper([], n_trials=1)
        study = optuna.create_study()
        sweeper._enqueue_initial_trials(study)
        assert study.trials == []


class TestResults:
    def _completed_study(self, make_sweeper, values, direction="minimize", **kw):
        sweeper = make_sweeper(values, n_trials=len(values), **kw)
        study = optuna.create_study(direction=direction)
        _run(sweeper, study, [direction])
        return sweeper, study

    def test_keeps_legacy_fields(self, make_sweeper):
        """Anything parsing optimization_results.yaml today must keep working."""
        sweeper, study = self._completed_study(make_sweeper, [5.0, 1.0, 9.0])
        results = sweeper._build_results(study, ["minimize"], 12.0)

        assert results["name"] == "optuna"
        assert results["best_value"] == 1.0
        assert "best_params" in results

    def test_reports_state_counts_and_elapsed(self, make_sweeper):
        sweeper = make_sweeper([5.0, optuna.TrialPruned(), "bad"], n_trials=3, max_failure_rate=1.0)
        study = optuna.create_study(direction="minimize")
        _run(sweeper, study, ["minimize"])

        results = sweeper._build_results(study, ["minimize"], 3725.0)

        assert results["trials"] == {"total": 3, "complete": 1, "fail": 1, "pruned": 1}
        assert results["elapsed"] == "1h 2m"

    def test_top_n_is_ordered_and_capped(self, make_sweeper):
        sweeper, study = self._completed_study(make_sweeper, [5.0, 1.0, 9.0, 3.0], results_top_n=2)
        results = sweeper._build_results(study, ["minimize"], 1.0)

        assert [t["value"] for t in results["top"]] == [1.0, 3.0]

    def test_top_n_follows_maximize(self, make_sweeper):
        sweeper, study = self._completed_study(
            make_sweeper, [5.0, 1.0, 9.0], direction="maximize", results_top_n=2
        )
        results = sweeper._build_results(study, ["maximize"], 1.0)

        assert [t["value"] for t in results["top"]] == [9.0, 5.0]

    def test_top_n_zero_omits_the_section(self, make_sweeper):
        sweeper, study = self._completed_study(make_sweeper, [5.0, 1.0], results_top_n=0)
        results = sweeper._build_results(study, ["minimize"], 1.0)

        assert "top" not in results

    def test_worker_time_summed_when_recorded(self, make_sweeper):
        sweeper = make_sweeper([], n_trials=0)
        study = optuna.create_study(direction="minimize")
        for i in range(2):
            study.add_trial(
                optuna.trial.create_trial(
                    params={},
                    distributions={},
                    value=float(i),
                    user_attrs={"worker_start": 100.0, "worker_end": 190.0},
                )
            )

        results = sweeper._build_results(study, ["minimize"], 3600.0)

        # 2 x 90s of real work inside an hour of wall clock.
        assert results["worker_time"] == "3m 0s"
        assert results["elapsed"] == "1h 0m"

    def test_worker_time_absent_without_timings(self, make_sweeper):
        sweeper, study = self._completed_study(make_sweeper, [5.0])
        results = sweeper._build_results(study, ["minimize"], 1.0)
        assert "worker_time" not in results

    def test_multi_objective_keeps_solutions(self, make_sweeper):
        sweeper = make_sweeper([[1.0, 2.0]], n_trials=1)
        study = optuna.create_study(directions=["minimize", "minimize"])
        _run(sweeper, study, ["minimize", "minimize"])

        results = sweeper._build_results(study, ["minimize", "minimize"], 60.0)

        assert results["solutions"] == [{"params": {}, "values": [1.0, 2.0]}]
        assert results["trials"]["complete"] == 1
        assert "top" not in results


class TestStorageDir:
    """SQLite will not create missing directories, and the natural place for the
    database — under hydra.sweep.dir — does not exist yet at create_study time."""

    def test_creates_missing_sqlite_directory(self, make_sweeper, tmp_path):
        sweeper = make_sweeper([], n_trials=1)
        target = tmp_path / "runs" / "my-sweep"
        sweeper.storage = f"sqlite:///{target}/study.db"

        sweeper._ensure_storage_dir()

        assert target.is_dir()

    def test_leaves_other_backends_alone(self, make_sweeper):
        sweeper = make_sweeper([], n_trials=1)
        sweeper.storage = "postgresql://user@host/db"
        sweeper._ensure_storage_dir()  # must not raise

    def test_handles_in_memory_and_none(self, make_sweeper):
        sweeper = make_sweeper([], n_trials=1)
        for url in ("sqlite:///:memory:", None):
            sweeper.storage = url
            sweeper._ensure_storage_dir()  # must not raise
