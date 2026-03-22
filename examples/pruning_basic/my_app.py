"""Pruning example: simulated training loop with intermediate reports."""

import optuna
import hydra
from omegaconf import DictConfig

from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    alpha: float = cfg.alpha
    beta: float = cfg.beta

    trial = get_current_trial()

    # Simulate a training loop with 10 epochs
    value = 0.0
    for step in range(10):
        # Synthetic objective that converges (or not) depending on params
        value = (alpha - 1.0) ** 2 + (beta - 0.5) ** 2 + step * 0.01

        if trial is not None:
            trial.report(value, step)
            if trial.should_prune():
                print(f"Trial pruned at step {step}, value={value:.4f}")
                raise optuna.TrialPruned()

    print(f"alpha={alpha:.4f}, beta={beta:.4f}, final_value={value:.4f}")
    return value


if __name__ == "__main__":
    main()
