import json
import logging
import os
import threading
import time

import optuna
from optuna.pruners import BasePruner
from optuna.trial import Trial

# Study user attribute carrying the sweeper's pruner config (_target_ + kwargs).
# Optuna does not persist the pruner itself in storage, so remote workers rebuild
# it from here instead of silently falling back to Optuna's default MedianPruner.
PRUNER_USER_ATTR = "_reborn_pruner"

_thread_local = threading.local()
_remote_trial_cache: Trial | None = None
_remote_trial_cache_key: tuple[str, str, str] | None = None

log = logging.getLogger(__name__)


class _TimedTrial(Trial):
    """Trial that records how long the worker actually ran.

    Optuna's ``datetime_start``/``datetime_complete`` are stamped by the sweep
    controller in ``ask()``/``tell()``, so they include the wait for the rest of
    the batch: a trial pruned after one epoch looks as long as its slowest
    neighbour. These user attributes carry the worker's own wall-clock instead.
    """

    def should_prune(self) -> bool:
        result = super().should_prune()
        # Called once per reported step, so the last write always lands on either
        # the moment of pruning or the end of training.
        self.set_user_attr("worker_end", time.time())
        return result


def set_current_trial(trial: Trial | None) -> None:
    """Set the current Optuna trial for this thread. Called by the sweeper."""
    _thread_local.current_trial = trial


def _load_pruner(study: optuna.Study) -> BasePruner | None:
    """Rebuild the sweeper's configured pruner from the study's user attributes."""
    payload = study.user_attrs.get(PRUNER_USER_ATTR)
    if not payload:
        return None

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    pruner = instantiate(OmegaConf.create(json.loads(payload)))
    if not isinstance(pruner, BasePruner):
        raise TypeError(f"Expected an Optuna BasePruner, got {type(pruner).__name__}")
    return pruner


def get_current_trial() -> Trial | None:
    """Get the current Optuna trial for this thread.

    Returns None if no trial is active (e.g., not running under the sweeper
    or pruning is not enabled).

    Supports two modes:
    - **Local (BasicLauncher)**: Trial is passed via thread-local storage.
    - **Remote (Ray/distributed)**: Trial is reconstructed from environment
      variables (OPTUNA_TRIAL_ID, OPTUNA_STUDY_NAME, OPTUNA_STORAGE) that the
      sweeper injects via ``hydra.job.env_set``. The configured pruner is read
      from the study's user attributes in the same storage.

    Usage in training code::

        from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial
        import optuna

        trial = get_current_trial()
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    """
    # 1. Thread-local (works with BasicLauncher in the same process)
    trial = getattr(_thread_local, "current_trial", None)
    if trial is not None:
        return trial

    # 2. Remote reconstruction from env vars (works with Ray/distributed)
    trial_id_str = os.environ.get("OPTUNA_TRIAL_ID")
    study_name = os.environ.get("OPTUNA_STUDY_NAME")
    storage_url = os.environ.get("OPTUNA_STORAGE")

    if trial_id_str is None or not study_name or not storage_url:
        return None

    # Return cached trial if it matches current env vars
    cache_key = (trial_id_str, study_name, storage_url)
    global _remote_trial_cache, _remote_trial_cache_key
    if _remote_trial_cache is not None and _remote_trial_cache_key == cache_key:
        return _remote_trial_cache

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        pruner = _load_pruner(study)
        if pruner is not None:
            study.pruner = pruner
        else:
            log.warning(
                f"No pruner config found in study '{study_name}' "
                f"(user attribute '{PRUNER_USER_ATTR}'); falling back to Optuna's "
                "default pruner. Pruning decisions may not match the sweeper's."
            )

        trial_id = int(trial_id_str)
        remote_trial = _TimedTrial(study, trial_id)
        remote_trial.set_user_attr("worker_start", time.time())
        _remote_trial_cache = remote_trial
        _remote_trial_cache_key = cache_key
        log.debug(
            f"Reconstructed trial {trial_id} from storage "
            f"(study={study_name}, storage={storage_url})"
        )
        return remote_trial
    except Exception as e:
        log.warning(f"Failed to reconstruct trial from env vars: {e}")
        return None


def clear_current_trial() -> None:
    """Clear the current trial. Called by the sweeper after job completion."""
    global _remote_trial_cache, _remote_trial_cache_key
    _thread_local.current_trial = None
    _remote_trial_cache = None
    _remote_trial_cache_key = None
