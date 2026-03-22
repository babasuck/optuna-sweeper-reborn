"""Dashboard example: sphere function with Optuna Dashboard visualization."""

import time

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    result = x**2 + y**2
    # Small delay so dashboard can update in real time
    time.sleep(0.5)
    print(f"x={x:.4f}, y={y:.4f}, result={result:.4f}")
    return result


if __name__ == "__main__":
    main()
