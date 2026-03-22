# hydra-optuna-sweeper-reborn

A modern, feature-rich [Optuna](https://optuna.org/) Sweeper plugin for [Hydra](https://hydra.cc/).

Drop-in replacement for the abandoned [hydra-optuna-sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/) with **pruning**, **dashboard**, **callbacks**, and full **Optuna 4.x** support.

---

## Why?

The original `hydra-optuna-sweeper` by Facebook/Meta is no longer maintained:

- Pinned to `optuna <3.0.0` and `sqlalchemy ~=1.3.0`
- No pruning support at all
- Uses deprecated Optuna distribution API
- Missing modern samplers (GP, NSGA-III, QMC, BruteForce)

**hydra-optuna-sweeper-reborn** fixes all of this while keeping backward-compatible configs.

---

## Installation

From GitHub:

```bash
pip install git+https://github.com/babasuck/optuna-sweeper-reborn.git
```

With dashboard support:

```bash
pip install "hydra-optuna-sweeper-reborn[dashboard] @ git+https://github.com/babasuck/optuna-sweeper-reborn.git"
```

---

## Quick Start

### 1. Basic sweep

```yaml
# config.yaml
defaults:
  - override /hydra/sweeper: optuna_reborn

x: 0.0
y: 0.0

hydra:
  mode: MULTIRUN
  sweeper:
    direction: minimize
    n_trials: 20
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    params:
      x: interval(-5.0, 5.0)
      y: interval(-5.0, 5.0)
```

```python
# my_app.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    return cfg.x ** 2 + cfg.y ** 2

if __name__ == "__main__":
    main()
```

```bash
python my_app.py -m
```

### 2. Sweep with pruning

The killer feature missing from the original plugin. Bad trials are stopped early, saving compute.

```yaml
# config.yaml
defaults:
  - override /hydra/sweeper: optuna_reborn

alpha: 1.0
beta: 0.5

hydra:
  mode: MULTIRUN
  sweeper:
    direction: minimize
    n_trials: 30
    enable_pruning: true
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 42
    pruner:
      _target_: optuna.pruners.MedianPruner
      n_startup_trials: 5
      n_warmup_steps: 3
    params:
      alpha: tag(log, interval(0.001, 100.0))
      beta: interval(0.0, 1.0)
```

In your training code, report intermediate values:

```python
from hydra_plugins.hydra_optuna_sweeper_reborn import get_current_trial
import optuna

trial = get_current_trial()
for epoch in range(max_epochs):
    val_loss = train_one_epoch(...)

    if trial is not None:
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

return val_loss
```

### 3. PyTorch Lightning integration

A ready-made callback is included in `examples/pruning_pytorch_lightning/optuna_pruning_callback.py`:

```python
from optuna_pruning_callback import OptunaPruningCallback

pruning_cb = OptunaPruningCallback(monitor="val_loss")
trainer = Trainer(callbacks=[pruning_cb, ...])
trainer.fit(model, datamodule)
return trainer.callback_metrics["val_loss"].item()
```

### 4. Multi-objective optimization

```yaml
defaults:
  - override /hydra/sweeper: optuna_reborn
  - override /hydra/sweeper/sampler: nsgaii

hydra:
  mode: MULTIRUN
  sweeper:
    direction:
      - minimize
      - minimize
    n_trials: 30
    sampler:
      seed: 42
      population_size: 20
    params:
      x: interval(0.0, 5.0)
      y: interval(0.0, 3.0)
```

Return a list from your function:

```python
return [objective_1, objective_2]
```

### 5. Dashboard + Callbacks

Auto-launch [optuna-dashboard](https://github.com/optuna/optuna-dashboard) for real-time visualization:

```yaml
hydra:
  sweeper:
    storage: "sqlite:///optuna_result.db"
    study_name: my-study
    dashboard:
      enabled: true
      port: 8080
    callbacks:
      - _target_: hydra_plugins.hydra_optuna_sweeper_reborn._callbacks.BestTrialCallback
```

Open `http://localhost:8080` in your browser while the sweep runs.

---

## Features at a glance

| Feature | Original plugin | Reborn |
|---|---|---|
| Optuna version | `<3.0.0` | `>=4.0.0` |
| Pruning | - | MedianPruner, HyperbandPruner, PercentilePruner, ThresholdPruner, PatientPruner, SuccessiveHalvingPruner, NopPruner |
| Dashboard | - | Auto-launch `optuna-dashboard` |
| Callbacks | - | Built-in + custom via `_target_` |
| Samplers | 6 (TPE, Random, CMA-ES, NSGA-II, MOTPE, Grid) | 9 (+ GP, NSGA-III, QMC, BruteForce) |
| Distribution API | Deprecated (`UniformDistribution`, etc.) | Modern (`FloatDistribution`, `IntDistribution`) |
| Ray launcher | Yes | Yes |
| Config compatibility | - | Partial (same fields, new sweeper name) |

---

## Configuration reference

### Sweeper config

```yaml
hydra:
  sweeper:
    # Same as original plugin
    direction: minimize          # or maximize, or [minimize, maximize] for multi-objective
    storage: null                # e.g. "sqlite:///optuna.db"
    study_name: null             # name for persistent studies
    n_trials: 20                 # total number of trials
    n_jobs: 2                    # parallel workers per batch
    max_failure_rate: 0.0        # 0.0 to 1.0

    sampler:                     # any Optuna sampler via _target_
      _target_: optuna.samplers.TPESampler
      seed: 42

    params:                      # search space
      lr: tag(log, interval(0.00001, 0.1))
      dropout: interval(0.0, 0.5)
      optimizer: choice(adam, sgd, adamw)

    # New in reborn
    enable_pruning: false        # enable trial pruning
    pruner:                      # any Optuna pruner via _target_
      _target_: optuna.pruners.MedianPruner
      n_startup_trials: 5
    dashboard:
      enabled: false
      host: localhost
      port: 8080
    callbacks:                   # list of callbacks via _target_
      - _target_: hydra_plugins.hydra_optuna_sweeper_reborn._callbacks.BestTrialCallback
```

### Available samplers (via defaults)

Override with `- override /hydra/sweeper/sampler: <name>`:

| Name | Sampler |
|---|---|
| `tpe` | TPESampler (default) |
| `random` | RandomSampler |
| `cmaes` | CmaEsSampler |
| `nsgaii` | NSGAIISampler |
| `nsgaiii` | NSGAIIISampler |
| `gp` | GPSampler |
| `qmc` | QMCSampler |
| `grid` | GridSampler |
| `bruteforce` | BruteForceSampler |

### Available pruners

Specify via `_target_` in the `pruner` config block:

| Pruner | `_target_` |
|---|---|
| MedianPruner | `optuna.pruners.MedianPruner` |
| HyperbandPruner | `optuna.pruners.HyperbandPruner` |
| PercentilePruner | `optuna.pruners.PercentilePruner` |
| ThresholdPruner | `optuna.pruners.ThresholdPruner` |
| PatientPruner | `optuna.pruners.PatientPruner` |
| SuccessiveHalvingPruner | `optuna.pruners.SuccessiveHalvingPruner` |
| NopPruner | `optuna.pruners.NopPruner` |

---

## Migrating from hydra-optuna-sweeper

Minimal changes required:

1. **Change the sweeper override:**
   ```yaml
   # Before
   - override /hydra/sweeper: optuna
   # After
   - override /hydra/sweeper: optuna_reborn
   ```

2. **Remove explicit `_target_`** from sweeper config (the defaults handle it):
   ```yaml
   # Remove this line
   _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
   ```

3. **Update dependencies:**
   ```
   # Before
   hydra-optuna-sweeper>=1.2.0
   optuna>=2.10.0,<3.0.0
   sqlalchemy==1.4.46

   # After
   hydra-optuna-sweeper-reborn @ git+https://github.com/babasuck/optuna-sweeper-reborn.git
   optuna>=4.0.0
   ```

All other config fields (`sampler`, `direction`, `storage`, `study_name`, `n_trials`, `n_jobs`, `params`) are **fully compatible** and work without changes. Ray launcher configs work as-is.

---

## Examples

See the [`examples/`](examples/) directory:

| Example | Description |
|---|---|
| [`simple/`](examples/simple/) | Basic sweep, minimize x^2 + y^2 |
| [`pruning_basic/`](examples/pruning_basic/) | Pruning with MedianPruner |
| [`pruning_pytorch_lightning/`](examples/pruning_pytorch_lightning/) | PyTorch Lightning + pruning callback |
| [`multi_objective/`](examples/multi_objective/) | Multi-objective with NSGA-II |
| [`dashboard/`](examples/dashboard/) | Dashboard + BestTrialCallback |

---

## Requirements

- Python >= 3.10
- Hydra >= 1.3.2
- Optuna >= 4.0.0
- OmegaConf >= 2.3.0

## License

MIT
