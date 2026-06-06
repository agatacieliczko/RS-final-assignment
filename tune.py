import argparse
import json
import os
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from SASRec import SASRec, SASRecConfig
from train import TrainDataset, eval_epoch


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter Tuning using Optuna")
    p.add_argument("--processed_dir", type=Path, default=Path("processed"))
    p.add_argument("--trials", type=int, default=30, help="Number of trials to run")
    p.add_argument("--epochs", type=int, default=15, help="Epochs per trial")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=69)
    return p.parse_args()


def objective(trial, train_loader, val_loader, num_items, num_categories, max_seq_len, item2cat, device, epochs):
    # suggest hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    
    # make sure that num_heads perfectly divides hidden_size
    possible_heads = [h for h in [2, 4, 8] if hidden_size % h == 0]
    num_heads = trial.suggest_categorical("num_heads", possible_heads)
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # 2. Build Model
    config = SASRecConfig(
        num_items=num_items,
        num_categories=num_categories,
        item2cat=item2cat,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate
    )
    model = SASRec(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # train and eval
    for epoch in range(epochs):
        model.train()
        for inp, pos, neg in train_loader:
            inp, pos, neg = inp.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            loss = model.calculate_loss(inp, pos, neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # evaluate
        val_loss = eval_epoch(model, val_loader, device)
        trial.report(val_loss, epoch)

        # prune based on the historical performance of other trials
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.processed_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    num_items = stats["num_items"]
    num_categories = stats.get("num_categories", 0)
    max_seq_len = stats["max_seq_len"]

    item2cat_path = args.processed_dir / "item2cat.npy"
    item2cat = np.load(item2cat_path).tolist() if item2cat_path.exists() else None

    # load data
    dataset = TrainDataset(args.processed_dir / "train_sasrec.npz", num_items)
    n_val = max(1, int(len(dataset) * 0.05))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    nw = min(4, os.cpu_count() or 1)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    study.optimize(lambda trial: objective(
        trial, train_loader, val_loader, num_items, num_categories, max_seq_len, item2cat, device, args.epochs
    ), n_trials=args.trials)

    print("\nBest trial:")
    print(f"  Value (Val Loss): {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

if __name__ == "__main__":
    main()