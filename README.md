# Recommender Systems Final Project

T# Recommender Systems Final Project

This project implements a sequential recommendation system using:

- SASRec (Self-Attentive Sequential Recommendation) for modeling sequential user behavior.
- Singular Value Decomposition (SVD) for capturing global collaborative filtering patterns.

The final submission model uses SASRec to rank candidate items for each user based on their historical interaction sequence.


## Installation

```bash
pip install -r requirements.txt
```

### Required Packages

- numpy
- pandas
- torch
- scipy
- scikit-learn
- optuna
- tqdm

## Step 1: Data Preprocessing

```bash
python preprocessing.py
```

## Step 2: Hyperparameter Tuning (Optional)

```bash
python tune.py --trials 30 --epochs 15
```

## Step 3: Train SASRec

```bash
python train.py
```

The best model checkpoint is saved as:

runs/exp/best_model.pt

## Step 4: Evaluate SASRec

```bash
python evaluate.py
```

Metrics reported:
- Recall@10
- NDCG@10

## Step 5: Generate Ensemble Submission

```bash
python ensemble_svd.py
```

This generates:

runs/exp/submission.csv

## Reproducibility

```bash
python preprocessing.py
python train.py
python ensemble_svd.py
```

## Authors

- Agata Cieliczko
- Patricija Dziuzaite
- Adam Mogyorosi
