from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from SASRec import SASRec, SASRecConfig


class EvalDataset(Dataset):
    """
    Each row: (input_seq, ground_truth_item_idx)
    Built from leave-one-out split of train interactions.
    """
    def __init__(
        self,
        seqs: np.ndarray,          # (N, max_seq_len)  input histories
        labels: np.ndarray,        # (N,)              ground-truth item idx (1-based)
    ):
        self.seqs   = torch.from_numpy(seqs.astype(np.int64))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def build_eval_data(
    train_df: pd.DataFrame,
    item2idx: dict[int, int],
    max_seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    
    # keep only users with ≥ 2 interactions
    counts = train_df.groupby("user_id").size()
    valid_users = counts[counts >= 2].index
    df = train_df[train_df["user_id"].isin(valid_users)].copy()
    df = df.sort_values(["user_id", "timestamp"])

    seqs, labels = [], []
    for uid, grp in df.groupby("user_id"):
        items = grp["item_id"].tolist()
        # Map to indices; skip any unmapped items
        idx_seq = [item2idx[i] for i in items if i in item2idx]
        if len(idx_seq) < 2:
            continue
        label   = idx_seq[-1]      # held-out ground truth
        history = idx_seq[:-1]     # input history

        # Right-pad
        history = history[-max_seq_len:]
        arr = np.zeros(max_seq_len, dtype=np.int64)
        arr[:len(history)] = history

        seqs.append(arr)
        labels.append(label)

    return np.vstack(seqs), np.array(labels, dtype=np.int64)


def recall_at_k(ranked: np.ndarray, label: int, k: int) -> float:
    return float(label in ranked[:k])


def ndcg_at_k(ranked: np.ndarray, label: int, k: int) -> float:
    for rank, item in enumerate(ranked[:k]):
        if item == label:
            return 1.0 / np.log2(rank + 2)
    return 0.0


@torch.no_grad()
def evaluate(
    model: SASRec,
    loader: DataLoader,
    device: torch.device,
    k: int = 10,
) -> dict[str, float]:
    model.eval()
    recall_sum, ndcg_sum, n = 0.0, 0.0, 0

    for seqs, labels in loader:
        seqs = seqs.to(device)
        # score all items (returns [B, num_items])
        scores = model.predict(seqs)                          # [B, num_items]
        # top-k indices (1-based item ids = column index + 1)
        topk   = torch.topk(scores, k, dim=-1).indices + 1   # [B, k]  1-based
        topk   = topk.cpu().numpy()
        labels = labels.numpy()

        for i in range(len(labels)):
            recall_sum += recall_at_k(topk[i], labels[i], k)
            ndcg_sum   += ndcg_at_k(topk[i], labels[i], k)
            n += 1

    return {
        f"Recall@{k}":  recall_sum / n,
        f"NDCG@{k}":    ndcg_sum   / n,
        "n_users":       n,
    }


TRAIN_CSV = "data/train.csv"
PROCESSED_DIR = "processed"
CHECKPOINT = "runs/exp/best_model.pt"  
BATCH_SIZE = 512
K = 10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed_dir = Path(PROCESSED_DIR)

    with open(processed_dir / "item2idx.json") as f:
        item2idx = {int(k): int(v) for k, v in json.load(f).items()}

    with open(processed_dir / "dataset_stats.json") as f:
        stats = json.load(f)

    num_items = stats["num_items"]
    max_seq_len = stats["max_seq_len"]

    config = SASRecConfig(
        num_items=num_items,
        max_seq_len=max_seq_len
    )

    model = SASRec(config).to(device)

    ckpt = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(ckpt["model_state_dict"])

    print(
        f"Loaded checkpoint: {CHECKPOINT} "
        f"(epoch {ckpt.get('epoch', '?')})"
    )



    train_df = pd.read_csv(TRAIN_CSV)

    seqs, labels = build_eval_data(
        train_df,
        item2idx,
        max_seq_len
    )

    eval_ds = EvalDataset(seqs, labels)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    metrics = evaluate(
        model,
        eval_loader,
        device,
        k=K
    )

    print("\n========== RESULTS ==========")
    print(f"Recall@{K}: {metrics[f'Recall@{K}']:.6f}")
    print(f"NDCG@{K}:   {metrics[f'NDCG@{K}']:.6f}")
    print(f"Users:      {metrics['n_users']}")
    print("=============================")


if __name__ == "__main__":
    main()