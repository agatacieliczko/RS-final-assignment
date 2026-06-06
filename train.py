from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

from SASRec import SASRec, SASRecConfig


class TrainDataset(Dataset):
    def __init__(self, npz_path: Path, num_items: int):
        d = np.load(npz_path)
        self.input_seq    = torch.from_numpy(d["input_seq"].astype(np.int64))
        self.positive_seq = torch.from_numpy(d["positive_seq"].astype(np.int64))
        self.num_items    = num_items

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        inp = self.input_seq[idx]
        pos = self.positive_seq[idx]
        neg = torch.zeros_like(pos)
        for i in range(len(pos)):
            if pos[i] != 0:
                # Randomly sample a negative item dynamically
                n = torch.randint(1, self.num_items + 1, (1,)).item()
                while n == pos[i]:
                    n = torch.randint(1, self.num_items + 1, (1,)).item()
                neg[i] = n
        return inp, pos, neg


class InferenceDataset(Dataset):
    def __init__(self, npz_path: Path):
        d = np.load(npz_path)
        self.ids             = torch.from_numpy(d["ID"].astype(np.int64))
        self.user_ids        = torch.from_numpy(d["user_id"].astype(np.int64))
        self.input_seq       = torch.from_numpy(d["input_seq"].astype(np.int64))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.user_ids[idx], self.input_seq[idx]


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for inp, pos, neg in tqdm(loader, desc="Train", leave=False):
        inp, pos, neg = inp.to(device), pos.to(device), neg.to(device)
        optimizer.zero_grad()
        loss = model.calculate_loss(inp, pos, neg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for inp, pos, neg in tqdm(loader, desc="Val  ", leave=False):
        inp, pos, neg = inp.to(device), pos.to(device), neg.to(device)
        total += model.calculate_loss(inp, pos, neg).item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def run_inference(model, loader, idx2item, device, out_path):
    model.eval()
    ids, users, ranked_strs = [], [], []

    for batch_ids, batch_users, batch_seq in tqdm(loader, desc="Inference"):
        batch_seq   = batch_seq.to(device)

        # Predict across ALL items in vocabulary
        scores      = model.predict(batch_seq)  # [B, num_items]
        topk_indices = torch.topk(scores, 10, dim=-1).indices + 1  # 1-based indexing alignment

        for i in range(len(batch_ids)):
            # Map predictions back to original item IDs
            ranked = [idx2item[str(idx)] for idx in topk_indices[i].cpu().tolist() if str(idx) in idx2item]
            ids.append(batch_ids[i].item())
            users.append(batch_users[i].item())
            ranked_strs.append(",".join(str(x) for x in ranked))

    df = pd.DataFrame({"ID": ids, "user_id": users, "item_id": ranked_strs})
    df.sort_values("ID").to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}  ({len(df)} rows)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, default=Path("processed"))
    p.add_argument("--output_dir",    type=Path, default=Path("runs/exp"))
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--val_split",     type=float, default=0.05)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--inference_only", action="store_true")
    p.add_argument("--checkpoint",    type=Path,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.processed_dir / "dataset_stats.json") as f:
        stats = json.load(f)
    with open(args.processed_dir / "idx2item.json") as f:
        idx2item = json.load(f)

    num_items   = stats["num_items"]
    num_categories = stats.get("num_categories", 0)
    max_seq_len = stats["max_seq_len"]

    # config = SASRecConfig(num_items=num_items, max_seq_len=max_seq_len)
    item2at_path = args.processed_dir / "item2cat.npy"
    item2cat = np.load(item2at_path).tolist() if item2at_path.exists() else None

    config = SASRecConfig(
        num_items=num_items,
        item2cat=item2cat,
        max_seq_len=max_seq_len,
        num_categories=num_categories,
    )
    model  = SASRec(config).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(config)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    inf_loader = DataLoader(
        InferenceDataset(args.processed_dir / "inference_sasrec.npz"),
        batch_size=512, shuffle=False,
        num_workers=min(4, os.cpu_count() or 1), pin_memory=torch.cuda.is_available(),
    )

    if args.inference_only:
        run_inference(model, inf_loader, idx2item, device, args.output_dir / "submission.csv")
        return

    # ---- Split train / val ----
    dataset = TrainDataset(args.processed_dir / "train_sasrec.npz", num_items)
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    nw = min(4, os.cpu_count() or 1)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=pin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val, no_improve = float("inf"), 0
    best_ckpt = args.output_dir / "best_model.pt"
    log_rows  = []

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={time.time()-t0:.1f}s"
        )
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "config": config}, best_ckpt)
            print(f"  ✓ Best checkpoint saved ({val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    pd.DataFrame(log_rows).to_csv(args.output_dir / "train_log.csv", index=False)

    print(f"\nLoading best checkpoint for inference…")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    run_inference(model, inf_loader, idx2item, device, args.output_dir / "submission.csv")


if __name__ == "__main__":
    main()