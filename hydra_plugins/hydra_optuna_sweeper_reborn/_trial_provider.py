import logging
import os
import pickle
import threading
from typing import Optional

from optuna.pruners import BasePruner
from optuna.trial import Trial

_thread_local = threading.local()
_remote_trial_cache: Optional[Trial] = None
_remote_trial_cache_key: Optional[tuple[str, str, str, Optional[str]]] = None

log = logging.getLogger(__name__)


def serialize_pruner(pruner: BasePruner) -> str:
    """Serialize a pruner for transport through Hydra's environment overrides."""
    return pickle.dumps(pruner, protocol=pickle.HIGHEST_PROTOCOL).hex()


def deserialize_pruner(payload: str) -> BasePruner:
    """Deserialize and validate a pruner received from the sweep controller."""
    pruner = pickle.loads(bytes.fromhex(payload))
    if not isinstance(pruner, BasePruner):
        raise TypeError(
            f"Expected an Optuna BasePruner, got {type(pruner).__name__}"
        )
    return pruner


def set_current_trial(trial: Optional[Trial]) -> None:
    """Set the current Optuna trial for this thread. Called by the sweeper."""
    _thread_local.current_trial = trial


def get_current_trial() -> Optional[Trial]:
    """Get the current Optuna trial for this thread.

    Returns None if no trial is active (e.g., not running under the sweeper
    or pruning is not enabled).

    Supports two modes:
    - **Local (BasicLauncher)**: Trial is passed via thread-local storage.
    - **Remote (Ray/distributed)**: Trial is reconstructed from environment
      variables (OPTUNA_TRIAL_ID, OPTUNA_STUDY_NAME, OPTUNA_STORAGE,
      OPTUNA_PRUNER) that the sweeper injects via ``hydra.job.env_set``.

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
    pruner_payload = os.environ.get("OPTUNA_PRUNER")

    if trial_id_str is None or not study_name or not storage_url:
        return None

    # Return cached trial if it matches current env vars
    cache_key = (trial_id_str, study_name, storage_url, pruner_payload)
    global _remote_trial_cache, _remote_trial_cache_key
    if _remote_trial_cache is not None and _remote_trial_cache_key == cache_key:
        return _remote_trial_cache

    try:
        import optuna

        pruner = deserialize_pruner(pruner_payload) if pruner_payload else None
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            pruner=pruner,
        )
        trial_id = int(trial_id_str)
        remote_trial = Trial(study, trial_id)
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
