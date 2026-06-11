from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader

from SASRec import SASRec, SASRecConfig
from train import InferenceDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROCESSED_DIR = Path("processed")
DATA_DIR = Path("data")
CHECKPOINT = Path("/kaggle/working/runs/exp/best_model.pt")OUT_PATH = Path("runs/exp/submission_ensemble_svd.csv")

SVD_COMPONENTS = 128
ALPHA = 0.6   # 0.8 = mostly SASRec, 0.2 = SVD
BATCH_SIZE = 512


def minmax_normalize(x):
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-8)


def build_svd_scores(num_items, item2idx):
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    full = pd.concat([train, test], ignore_index=True)

    users = sorted(full["user_id"].unique())
    user2row = {u: i for i, u in enumerate(users)}

    rows = full["user_id"].map(user2row).to_numpy()
    cols = full["item_id"].map(item2idx).to_numpy() - 1
    vals = np.ones(len(full), dtype=np.float32)

    X = csr_matrix((vals, (rows, cols)), shape=(len(users), num_items))

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_.T

    return user2row, X, user_factors, item_factors


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PROCESSED_DIR / "dataset_stats.json") as f:
        stats = json.load(f)

    with open(PROCESSED_DIR / "item2idx.json") as f:
        item2idx = {int(k): int(v) for k, v in json.load(f).items()}

    with open(PROCESSED_DIR / "idx2item.json") as f:
        idx2item = {int(k): int(v) for k, v in json.load(f).items()}

    num_items = stats["num_items"]
    max_seq_len = stats["max_seq_len"]

    item2cat_path = PROCESSED_DIR / "item2cat.npy"
    item2cat = np.load(item2cat_path).tolist() if item2cat_path.exists() else None
    num_categories = stats.get("num_categories", 0)

    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = ckpt.get(
        "config",
        SASRecConfig(
            num_items=num_items,
            max_seq_len=max_seq_len,
            num_categories=num_categories,
            item2cat=item2cat,
        ),
    )

    model = SASRec(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    user2row, X, user_factors, item_factors = build_svd_scores(num_items, item2idx)

    loader = DataLoader(
        InferenceDataset(PROCESSED_DIR / "inference_sasrec.npz"),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ids, users_out, ranked_strs = [], [], []

    for batch_ids, batch_users, batch_seq in loader:
        batch_seq = batch_seq.to(device)

        sas_scores = model.predict(batch_seq).cpu().numpy()
        sas_scores = minmax_normalize(sas_scores)

        svd_scores = np.zeros_like(sas_scores, dtype=np.float32)

        for b, user_id in enumerate(batch_users.numpy()):
            if int(user_id) in user2row:
                row = user2row[int(user_id)]
                svd_scores[b] = user_factors[row] @ item_factors.T

        svd_scores = minmax_normalize(svd_scores)

        scores = ALPHA * sas_scores + (1.0 - ALPHA) * svd_scores

        for b in range(batch_seq.size(0)):
            seen = batch_seq[b].cpu().numpy()
            seen_idx = seen[seen > 0] - 1
            scores[b, seen_idx] = -np.inf

        topk = np.argpartition(-scores, kth=10, axis=1)[:, :10]

        for b in range(topk.shape[0]):
            ordered = topk[b][np.argsort(-scores[b, topk[b]])] + 1
            ranked_items = [idx2item[int(i)] for i in ordered]

            ids.append(int(batch_ids[b]))
            users_out.append(int(batch_users[b]))
            ranked_strs.append(",".join(str(x) for x in ranked_items))

    sub = pd.DataFrame({
        "ID": ids,
        "user_id": users_out,
        "item_id": ranked_strs,
    })

    sub.sort_values("ID").to_csv(OUT_PATH, index=False)
    print(f"Saved ensemble submission to {OUT_PATH}")


if __name__ == "__main__":
    main()
