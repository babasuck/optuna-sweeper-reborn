import json
import os
import threading

import optuna
import pytest

from hydra_plugins.hydra_optuna_sweeper_reborn._trial_provider import (
    PRUNER_USER_ATTR,
    clear_current_trial,
    get_current_trial,
    set_current_trial,
)

REMOTE_ENV_VARS = (
    "OPTUNA_TRIAL_ID",
    "OPTUNA_STUDY_NAME",
    "OPTUNA_STORAGE",
)


class TestTrialProvider:
    def test_get_current_trial_default_none(self):
        clear_current_trial()
        assert get_current_trial() is None

    def test_set_and_get_trial(self):
        sentinel = object()
        set_current_trial(sentinel)
        assert get_current_trial() is sentinel
        clear_current_trial()

    def test_clear_trial(self):
        set_current_trial(object())
        clear_current_trial()
        assert get_current_trial() is None

    def test_thread_isolation(self):
        """Each thread should have its own trial."""
        results = {}

        def worker(name, trial_obj):
            set_current_trial(trial_obj)
            results[name] = get_current_trial()
            clear_current_trial()

        obj_a = object()
        obj_b = object()

        t1 = threading.Thread(target=worker, args=("a", obj_a))
        t2 = threading.Thread(target=worker, args=("b", obj_b))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"] is obj_a
        assert results["b"] is obj_b


class TestTrialProviderEnvVarFallback:
    """Test that get_current_trial() falls back to env vars for remote workers."""

    def setup_method(self):
        clear_current_trial()
        # Clean env vars
        for key in REMOTE_ENV_VARS:
            os.environ.pop(key, None)

    def teardown_method(self):
        clear_current_trial()
        for key in REMOTE_ENV_VARS:
            os.environ.pop(key, None)

    @staticmethod
    def _set_remote_trial_env(study, trial, storage_url):
        os.environ["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
        os.environ["OPTUNA_STUDY_NAME"] = study.study_name
        os.environ["OPTUNA_STORAGE"] = storage_url

    @staticmethod
    def _publish_pruner(study, config):
        """Mirror of OptunaSweeperImpl._publish_pruner_config."""
        study.set_user_attr(PRUNER_USER_ATTR, json.dumps(config))

    def test_no_env_vars_returns_none(self):
        assert get_current_trial() is None

    def test_partial_env_vars_returns_none(self):
        os.environ["OPTUNA_TRIAL_ID"] = "1"
        # Missing STUDY_NAME and STORAGE
        assert get_current_trial() is None

    def test_env_var_fallback_reconstructs_trial(self, tmp_path):
        """Full env var fallback: create a study, ask a trial, then reconstruct."""
        storage_url = f"sqlite:///{tmp_path}/test.db"
        study = optuna.create_study(study_name="test-env", storage=storage_url)
        trial = study.ask()

        self._set_remote_trial_env(study, trial, storage_url)

        reconstructed = get_current_trial()
        assert reconstructed is not None
        assert reconstructed._trial_id == trial._trial_id

        # Verify report() works on the reconstructed trial
        reconstructed.report(0.5, step=0)
        reconstructed.report(0.3, step=1)

        # Verify intermediate values were stored
        frozen = study._storage.get_trial(trial._trial_id)
        assert frozen.intermediate_values[0] == 0.5
        assert frozen.intermediate_values[1] == 0.3

    def test_env_var_should_prune_works(self, tmp_path):
        """Verify should_prune() works on reconstructed trial."""
        storage_url = f"sqlite:///{tmp_path}/test_prune.db"
        study = optuna.create_study(
            study_name="test-prune",
            storage=storage_url,
            pruner=optuna.pruners.NopPruner(),
        )
        trial = study.ask()

        self._set_remote_trial_env(study, trial, storage_url)
        self._publish_pruner(study, {"_target_": "optuna.pruners.NopPruner"})

        reconstructed = get_current_trial()
        reconstructed.report(0.5, step=0)

        # NopPruner never prunes
        assert reconstructed.should_prune() is False

    def test_configured_median_pruner_is_preserved_on_remote_worker(self, tmp_path):
        """Remote workers must not fall back to MedianPruner(5, 0)."""
        storage_url = f"sqlite:///{tmp_path}/test_median_pruner.db"
        pruner_config = {
            "_target_": "optuna.pruners.MedianPruner",
            "n_startup_trials": 20,
            "n_warmup_steps": 5,
            "interval_steps": 1,
        }
        study = optuna.create_study(
            study_name="test-median-pruner",
            storage=storage_url,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=20, n_warmup_steps=5, interval_steps=1
            ),
        )
        self._publish_pruner(study, pruner_config)

        # Five completed trials are enough for Optuna's default MedianPruner to
        # prune at step 0, but not for the configured 20-trial/5-step warmup.
        for _ in range(5):
            baseline = study.ask()
            baseline.report(1.0, step=0)
            baseline.report(1.0, step=5)
            study.tell(baseline, 1.0)

        trial = study.ask()
        self._set_remote_trial_env(study, trial, storage_url)

        reconstructed = get_current_trial()
        reconstructed.report(0.0, step=0)

        assert reconstructed.should_prune() is False

        # Once 20 trials and five warmup steps are available, the same poor
        # trial should become eligible for pruning.
        for _ in range(15):
            baseline = study.ask()
            baseline.report(1.0, step=0)
            baseline.report(1.0, step=5)
            study.tell(baseline, 1.0)

        reconstructed.report(0.0, step=5)

        assert bool(reconstructed.should_prune()) is True

    def test_pruner_config_travels_via_study_user_attrs(self, tmp_path):
        """The pruner config rides in storage, not in an env var."""
        storage_url = f"sqlite:///{tmp_path}/test_attrs.db"
        study = optuna.create_study(study_name="test-attrs", storage=storage_url)
        self._publish_pruner(
            study,
            {
                "_target_": "optuna.pruners.PatientPruner",
                "patience": 3,
                "wrapped_pruner": {
                    "_target_": "optuna.pruners.MedianPruner",
                    "n_startup_trials": 9,
                },
            },
        )
        trial = study.ask()
        self._set_remote_trial_env(study, trial, storage_url)

        reconstructed = get_current_trial()

        # Nested pruners survive the round trip, and no OPTUNA_PRUNER env var
        # is involved at any point.
        assert "OPTUNA_PRUNER" not in os.environ
        assert isinstance(reconstructed.study.pruner, optuna.pruners.PatientPruner)
        assert isinstance(reconstructed.study.pruner._wrapped_pruner, optuna.pruners.MedianPruner)
        assert reconstructed.study.pruner._wrapped_pruner._n_startup_trials == 9

    def test_worker_timings_recorded_on_remote_trial(self, tmp_path):
        """worker_start/worker_end give the worker's own wall-clock, unlike
        datetime_start/complete which are stamped by the controller."""
        storage_url = f"sqlite:///{tmp_path}/test_timing.db"
        study = optuna.create_study(
            study_name="test-timing",
            storage=storage_url,
            pruner=optuna.pruners.NopPruner(),
        )
        self._publish_pruner(study, {"_target_": "optuna.pruners.NopPruner"})
        trial = study.ask()
        self._set_remote_trial_env(study, trial, storage_url)

        reconstructed = get_current_trial()
        reconstructed.report(0.5, step=0)
        reconstructed.should_prune()

        attrs = study._storage.get_trial(trial._trial_id).user_attrs
        assert "worker_start" in attrs
        assert "worker_end" in attrs
        assert attrs["worker_end"] >= attrs["worker_start"]

    def test_thread_local_takes_precedence_over_env(self, tmp_path):
        """Thread-local trial should take precedence over env vars."""
        storage_url = f"sqlite:///{tmp_path}/test_precedence.db"
        study = optuna.create_study(study_name="test-prec", storage=storage_url)
        trial = study.ask()

        self._set_remote_trial_env(study, trial, storage_url)

        sentinel = object()
        set_current_trial(sentinel)

        # Thread-local should win
        assert get_current_trial() is sentinel
        clear_current_trial()

    def test_stale_cache_is_not_reused_for_a_different_trial(self, tmp_path):
        """The cache is keyed by trial id, so a reused worker process does not
        hand back the previous trial."""
        storage_url = f"sqlite:///{tmp_path}/test_cache.db"
        study = optuna.create_study(study_name="test-cache", storage=storage_url)
        first, second = study.ask(), study.ask()

        self._set_remote_trial_env(study, first, storage_url)
        assert get_current_trial()._trial_id == first._trial_id

        self._set_remote_trial_env(study, second, storage_url)
        assert get_current_trial()._trial_id == second._trial_id


@pytest.mark.parametrize("payload", ['{"_target_": "builtins.dict"}'])
def test_non_pruner_target_is_rejected(tmp_path, payload, monkeypatch):
    """A config that does not build a BasePruner must not be silently accepted."""
    storage_url = f"sqlite:///{tmp_path}/test_bad.db"
    study = optuna.create_study(study_name="test-bad", storage=storage_url)
    study.set_user_attr(PRUNER_USER_ATTR, payload)
    trial = study.ask()

    monkeypatch.setenv("OPTUNA_TRIAL_ID", str(trial._trial_id))
    monkeypatch.setenv("OPTUNA_STUDY_NAME", study.study_name)
    monkeypatch.setenv("OPTUNA_STORAGE", storage_url)
    clear_current_trial()

    # get_current_trial swallows the failure and reports no trial rather than
    # handing back something with a bogus pruner.
    assert get_current_trial() is None
    clear_current_trial()
