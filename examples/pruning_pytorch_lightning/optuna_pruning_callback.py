"""Ready-made PyTorch Lightning callback for Optuna pruning integration."""

import optuna

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback


class OptunaPruningCallback(Callback):
    """PyTorch Lightning callback that reports metrics to Optuna and prunes bad trials.

    Usage::

        from optuna_pruning_callback import OptunaPruningCallback

        callbacks = [
            OptunaPruningCallback(monitor="val_loss"),
            # ... other callbacks
        ]
        trainer = Trainer(callbacks=callbacks)
    """

    def __init__(self, monitor: str = "val_loss") -> None:
        self.monitor = monitor
        self._trial = None

    def on_fit_start(self, trainer, pl_module):
        from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial

        self._trial = get_current_trial()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._trial is None:
            return

        current_value = trainer.callback_metrics.get(self.monitor)
        if current_value is None:
            return

        epoch = trainer.current_epoch
        self._trial.report(current_value.item(), epoch)

        if self._trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with {self.monitor}={current_value:.4f}"
            )
