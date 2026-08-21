import os
import threading

import optuna

from hydra_plugins.hydra_optuna_sweeper_reborn._trial_provider import (
    clear_current_trial,
    deserialize_pruner,
    get_current_trial,
    serialize_pruner,
    set_current_trial,
)


REMOTE_ENV_VARS = (
    "OPTUNA_TRIAL_ID",
    "OPTUNA_STUDY_NAME",
    "OPTUNA_STORAGE",
    "OPTUNA_PRUNER",
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
    def _set_remote_trial_env(study, trial, storage_url, pruner=None):
        os.environ["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
        os.environ["OPTUNA_STUDY_NAME"] = study.study_name
        os.environ["OPTUNA_STORAGE"] = storage_url
        if pruner is not None:
            os.environ["OPTUNA_PRUNER"] = serialize_pruner(pruner)

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

        self._set_remote_trial_env(study, trial, storage_url, study.pruner)

        reconstructed = get_current_trial()
        reconstructed.report(0.5, step=0)

        # NopPruner never prunes
        assert reconstructed.should_prune() is False

    def test_configured_median_pruner_is_preserved_on_remote_worker(self, tmp_path):
        """Remote workers must not fall back to MedianPruner(5, 0)."""
        storage_url = f"sqlite:///{tmp_path}/test_median_pruner.db"
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=5,
            interval_steps=1,
        )
        study = optuna.create_study(
            study_name="test-median-pruner",
            storage=storage_url,
            direction="maximize",
            pruner=pruner,
        )

        # Five completed trials are enough for Optuna's default MedianPruner to
        # prune at step 0, but not for the configured 20-trial/5-step warmup.
        for _ in range(5):
            baseline = study.ask()
            baseline.report(1.0, step=0)
            baseline.report(1.0, step=5)
            study.tell(baseline, 1.0)

        trial = study.ask()
        self._set_remote_trial_env(study, trial, storage_url, pruner)

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

    def test_pruner_serialization_roundtrip(self):
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=5,
        )

        restored = deserialize_pruner(serialize_pruner(pruner))

        assert isinstance(restored, optuna.pruners.MedianPruner)
        assert restored._n_startup_trials == 20
        assert restored._n_warmup_steps == 5

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
