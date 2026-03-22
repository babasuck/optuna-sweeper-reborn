# hydra-optuna-sweeper-reborn

An updated Optuna Sweeper plugin for Hydra with pruning support, dashboard integration, and modern Optuna 4.x compatibility.

## Installation

```bash
pip install hydra-optuna-sweeper-reborn
```

With dashboard support:
```bash
pip install hydra-optuna-sweeper-reborn[dashboard]
```

## Quick Start

```yaml
defaults:
  - override /hydra/sweeper: optuna_reborn

hydra:
  mode: MULTIRUN
  sweeper:
    direction: maximize
    n_trials: 50
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    params:
      model.lr: tag(log, interval(0.00001, 0.1))
```

## Features

- Pruning support (MedianPruner, HyperbandPruner, PercentilePruner, ThresholdPruner, PatientPruner)
- Optuna Dashboard auto-launch
- Study callbacks (progress logging, best trial notifications, custom)
- Modern Optuna 4.x API
- Backward-compatible config with the original hydra-optuna-sweeper
