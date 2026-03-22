import threading

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
