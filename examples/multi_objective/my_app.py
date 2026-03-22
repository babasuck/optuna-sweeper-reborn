"""Multi-objective example: minimize two conflicting objectives."""

from typing import List

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> List[float]:
    x: float = cfg.x
    y: float = cfg.y
    # Two conflicting objectives (ZDT1-like)
    f1 = x
    f2 = (1 + y) * (1 - (x / (1 + y)) ** 0.5)
    print(f"x={x:.4f}, y={y:.4f}, f1={f1:.4f}, f2={f2:.4f}")
    return [f1, f2]


if __name__ == "__main__":
    main()
