import os

import torch


def main():
    train_X = torch.load("data/train_X.pt")
    mean = torch.mean(train_X)
    std = torch.std(train_X)
    print(f"mean = {mean}, std = {std}")


if __name__ == "__main__":
    main()
