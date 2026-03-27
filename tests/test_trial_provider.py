import os
import threading

import optuna
import pytest

from hydra_plugins.hydra_optuna_sweeper_reborn._trial_provider import (
    clear_current_trial,
    get_current_trial,
    set_current_trial,
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
        for key in ("OPTUNA_TRIAL_ID", "OPTUNA_STUDY_NAME", "OPTUNA_STORAGE"):
            os.environ.pop(key, None)

    def teardown_method(self):
        clear_current_trial()
        for key in ("OPTUNA_TRIAL_ID", "OPTUNA_STUDY_NAME", "OPTUNA_STORAGE"):
            os.environ.pop(key, None)

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

        os.environ["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
        os.environ["OPTUNA_STUDY_NAME"] = "test-env"
        os.environ["OPTUNA_STORAGE"] = storage_url

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

        os.environ["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
        os.environ["OPTUNA_STUDY_NAME"] = "test-prune"
        os.environ["OPTUNA_STORAGE"] = storage_url

        reconstructed = get_current_trial()
        reconstructed.report(0.5, step=0)

        # NopPruner never prunes
        assert reconstructed.should_prune() is False

    def test_thread_local_takes_precedence_over_env(self, tmp_path):
        """Thread-local trial should take precedence over env vars."""
        storage_url = f"sqlite:///{tmp_path}/test_precedence.db"
        study = optuna.create_study(study_name="test-prec", storage=storage_url)
        trial = study.ask()

        os.environ["OPTUNA_TRIAL_ID"] = str(trial._trial_id)
        os.environ["OPTUNA_STUDY_NAME"] = "test-prec"
        os.environ["OPTUNA_STORAGE"] = storage_url

        sentinel = object()
        set_current_trial(sentinel)

        # Thread-local should win
        assert get_current_trial() is sentinel
        clear_current_trial()
