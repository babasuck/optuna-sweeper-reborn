"""Example: PyTorch Lightning training with Optuna pruning.

This is a minimal example showing how to integrate the OptunaPruningCallback.
Requires: pip install lightning torch
"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    # This is a simplified example. In a real project you would:
    # 1. Create your LightningModule
    # 2. Create your DataModule
    # 3. Add OptunaPruningCallback to trainer callbacks
    # 4. Train and return the metric

    # Example pseudocode:
    # from optuna_pruning_callback import OptunaPruningCallback
    #
    # model = MyModel(lr=cfg.lr, weight_decay=cfg.weight_decay)
    # datamodule = MyDataModule()
    # pruning_callback = OptunaPruningCallback(monitor="val_loss")
    # trainer = Trainer(
    #     max_epochs=50,
    #     callbacks=[pruning_callback, ...],
    # )
    # trainer.fit(model, datamodule)
    # return trainer.callback_metrics["val_loss"].item()

    # Simulated training for demonstration
    import math
    import optuna
    from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial

    lr = cfg.lr
    weight_decay = cfg.weight_decay
    trial = get_current_trial()

    # Simulate loss convergence
    best_val_loss = float("inf")
    for epoch in range(20):
        # Synthetic val_loss that depends on hyperparameters
        val_loss = (
            math.log(lr + 1e-8) ** 2 * 0.1
            + weight_decay * 10
            + 1.0 / (epoch + 1)
            + 0.1
        )
        best_val_loss = min(best_val_loss, val_loss)

        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"Pruned at epoch {epoch}, val_loss={val_loss:.4f}")
                raise optuna.TrialPruned()

    print(f"lr={lr:.6f}, wd={weight_decay:.6f}, val_loss={best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    main()
