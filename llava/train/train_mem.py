from pathlib import Path
import sys
path1 = str(Path(__file__).joinpath("../../..").resolve())
sys.path.append(path1)

from llava.train.train import train

import wandb
wandb.init(mode="disabled")


if __name__ == "__main__":
    train()
