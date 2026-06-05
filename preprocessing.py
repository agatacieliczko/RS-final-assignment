from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


REQUIRED_INTERACTION_COLUMNS = {"user_id", "item_id", "timestamp"}


def read_interactions(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_INTERACTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    df = df[["user_id", "item_id", "timestamp"]].copy()
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def build_item_mapping(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_meta_path: Path | None,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Create contiguous item ids. 0 is reserved for padding."""
    item_ids: Set[int] = set(train_df["item_id"].unique()) | set(test_df["item_id"].unique())

    if item_meta_path is not None and item_meta_path.exists():
        meta = pd.read_csv(item_meta_path, usecols=["item_id"])
        item_ids |= set(meta["item_id"].dropna().astype(int).unique())

    sorted_items = sorted(item_ids)
    item2idx = {item_id: idx + 1 for idx, item_id in enumerate(sorted_items)}
    idx2item = {idx: item_id for item_id, idx in item2idx.items()}
    return item2idx, idx2item


def map_items(df: pd.DataFrame, item2idx: Dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["item_idx"] = out["item_id"].map(item2idx)
    if out["item_idx"].isna().any():
        bad = out.loc[out["item_idx"].isna(), "item_id"].unique()[:10]
        raise ValueError(f"Found item IDs without mapping, examples: {bad}")
    out["item_idx"] = out["item_idx"].astype(int)
    return out


def right_pad(seq: Sequence[int], max_seq_len: int) -> np.ndarray:
    """Truncate to the most recent max_seq_len values, then right-pad with 0."""
    seq = list(seq)[-max_seq_len:]
    arr = np.zeros(max_seq_len, dtype=np.int64)
    arr[: len(seq)] = seq
    return arr


def sample_negative(num_items: int, forbidden: Set[int], rng: random.Random) -> int:
    """Sample one item index not in forbidden. Item ids are 1..num_items."""
    if len(forbidden) >= num_items:
        raise ValueError("Cannot negative-sample: user has interacted with all items.")

    neg = rng.randint(1, num_items)
    while neg in forbidden:
        neg = rng.randint(1, num_items)
    return neg


def build_user_histories(df: pd.DataFrame) -> Dict[int, List[int]]:
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    return df.groupby("user_id")["item_idx"].apply(list).to_dict()


def build_training_arrays(
    train_histories: Dict[int, List[int]],
    all_seen_by_user: Dict[int, Set[int]],
    num_items: int,
    max_seq_len: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng = random.Random(seed)
    user_ids: List[int] = []
    input_rows: List[np.ndarray] = []
    positive_rows: List[np.ndarray] = []
    negative_rows: List[np.ndarray] = []

    for user_id, items in train_histories.items():
        if len(items) < 2:
            continue

        inputs = items[:-1]
        positives = items[1:]

        # Keep most recent transitions only.
        inputs = inputs[-max_seq_len:]
        positives = positives[-max_seq_len:]

        input_arr = right_pad(inputs, max_seq_len)
        pos_arr = right_pad(positives, max_seq_len)
        neg_arr = np.zeros(max_seq_len, dtype=np.int64)

        forbidden = all_seen_by_user.get(user_id, set())
        for pos_idx, pos_item in enumerate(pos_arr):
            if pos_item != 0:
                neg_arr[pos_idx] = sample_negative(num_items, forbidden, rng)

        user_ids.append(user_id)
        input_rows.append(input_arr)
        positive_rows.append(pos_arr)
        negative_rows.append(neg_arr)

    return (
        np.asarray(user_ids, dtype=np.int64),
        np.vstack(input_rows).astype(np.int64),
        np.vstack(positive_rows).astype(np.int64),
        np.vstack(negative_rows).astype(np.int64),
    )


def build_inference_arrays(
    full_df: pd.DataFrame,
    sample_submission_path: Path,
    item2idx: Dict[int, int],
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    sub = pd.read_csv(sample_submission_path)
    if "user_id" not in sub.columns:
        raise ValueError("sample_submission must contain a user_id column")

    id_values = sub["ID"].to_numpy(dtype=np.int64) if "ID" in sub.columns else np.arange(len(sub))
    user_values = sub["user_id"].astype(int).to_numpy(dtype=np.int64)

    histories = build_user_histories(full_df)
    seq_rows: List[np.ndarray] = []
    candidate_rows: List[np.ndarray] = []

    for i, user_id in enumerate(user_values):
        seq_rows.append(right_pad(histories.get(int(user_id), []), max_seq_len))

        # map the 10 candidate item_ids -> contiguous indices
        raw_candidates = str(sub["item_id"].iloc[i]).split(",")
        mapped = [item2idx[int(c.strip())] for c in raw_candidates]
        candidate_rows.append(np.asarray(mapped, dtype=np.int64))

    return (
        id_values,
        user_values,
        np.vstack(seq_rows).astype(np.int64),
        np.vstack(candidate_rows).astype(np.int64),
    )


def save_json(path: Path, obj: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess data for SASRec.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--item_meta", type=Path, default=Path("data/item_meta.csv"))
    parser.add_argument("--sample_submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("processed"))
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = read_interactions(args.train, "train")
    test_df = read_interactions(args.test, "test")

    item_meta_path = args.item_meta if args.item_meta.exists() else None
    item2idx, idx2item = build_item_mapping(train_df, test_df, item_meta_path)
    num_items = len(item2idx)

    train_df = map_items(train_df, item2idx)
    test_df = map_items(test_df, item2idx)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    train_histories = build_user_histories(train_df)
    full_histories = build_user_histories(full_df)
    all_seen_by_user = {u: set(items) for u, items in full_histories.items()}

    user_id, input_seq, positive_seq, negative_seq = build_training_arrays(
        train_histories=train_histories,
        all_seen_by_user=all_seen_by_user,
        num_items=num_items,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )

    np.savez_compressed(
        args.output_dir / "train_sasrec.npz",
        user_id=user_id,
        input_seq=input_seq,
        positive_seq=positive_seq,
        negative_seq=negative_seq,
    )

    inf_id, inf_user_id, inf_input_seq, inf_candidates = build_inference_arrays(
        full_df=full_df,
        sample_submission_path=args.sample_submission,
        item2idx=item2idx,
        max_seq_len=args.max_seq_len,
    )
    np.savez_compressed(
        args.output_dir / "inference_sasrec.npz",
        ID=inf_id,
        user_id=inf_user_id,
        input_seq=inf_input_seq,
        candidate_items=inf_candidates,
    )

    save_json(args.output_dir / "item2idx.json", {str(k): int(v) for k, v in item2idx.items()})
    save_json(args.output_dir / "idx2item.json", {str(k): int(v) for k, v in idx2item.items()})

    stats = {
        "num_items": num_items,
        "max_seq_len": args.max_seq_len,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "num_train_users": int(train_df["user_id"].nunique()),
        "num_full_users": int(full_df["user_id"].nunique()),
        "num_training_examples": int(len(user_id)),
        "num_inference_users": int(len(inf_user_id)),
        "padding_index": 0,
        "sequence_padding": "right",
    }
    save_json(args.output_dir / "dataset_stats.json", stats)

    print("Preprocessing complete.")
    print(json.dumps(stats, indent=2))
    print(f"Saved files to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()