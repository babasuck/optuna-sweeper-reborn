import threading
from typing import Optional

from optuna.trial import Trial

_thread_local = threading.local()


def set_current_trial(trial: Optional[Trial]) -> None:
    """Set the current Optuna trial for this thread. Called by the sweeper."""
    _thread_local.current_trial = trial


def get_current_trial() -> Optional[Trial]:
    """Get the current Optuna trial for this thread.

    Returns None if no trial is active (e.g., not running under the sweeper
    or pruning is not enabled).

    Usage in training code::

        from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial
        import optuna

        trial = get_current_trial()
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    """
    return getattr(_thread_local, "current_trial", None)


def clear_current_trial() -> None:
    """Clear the current trial. Called by the sweeper after job completion."""
    _thread_local.current_trial = None
