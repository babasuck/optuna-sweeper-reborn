"""PyTorch Lightning training with Optuna pruning.

Trains a tiny regressor on synthetic tensors — no datasets to download — so the
focus stays on wiring OptunaPruningCallback into a Trainer.

Requires: pip install lightning torch
"""

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from optuna_pruning_callback import OptunaPruningCallback
from torch.utils.data import DataLoader, TensorDataset


class Regressor(L.LightningModule):
    def __init__(self, lr: float, weight_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.net(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.net(x), y)
        self.log("val_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def make_loaders(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(16, 1, generator=g)
    x = torch.randn(512, 16, generator=g)
    y = x @ w + 0.1 * torch.randn(512, 1, generator=g)
    train = TensorDataset(x[:384], y[:384])
    val = TensorDataset(x[384:], y[384:])
    return DataLoader(train, batch_size=32), DataLoader(val, batch_size=64)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    torch.manual_seed(0)
    train_loader, val_loader = make_loaders()

    model = Regressor(lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="cpu",
        callbacks=[OptunaPruningCallback(monitor="val_loss")],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # OptunaPruningCallback raises optuna.TrialPruned; the sweeper catches it and
    # records the trial as PRUNED, so let it propagate.
    trainer.fit(model, train_loader, val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()
    print(f"lr={cfg.lr:.6f}, wd={cfg.weight_decay:.6f}, val_loss={val_loss:.4f}")
    return val_loss


if __name__ == "__main__":
    main()
