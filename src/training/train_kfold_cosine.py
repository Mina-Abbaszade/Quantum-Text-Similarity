import os
import pickle
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.models.cosine_model import CosineSimilarityModel
from src.data.pair_dataset import PairDataset, collate_fn
from src.training.losses import accuracy, contrastive_similarity_loss

from lambeq import PytorchTrainer

# CONFIG
SEED = 1357408344
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.000345
K = 5

DATA_DIR = "data/processed"
CIRCUIT_PATH = "data/circuits/circuits.pkl"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# LOAD DATA
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

with open(CIRCUIT_PATH, "rb") as f:
    circuits_with_ids = pickle.load(f)

circuits_dict = {id_: circuit for id_, circuit in circuits_with_ids}

def generate_pair_circuits(df):
    pairs, labels = [], []

    for _, row in df.iterrows():
        id1, id2, label = row["id1"], row["id2"], row["labels"]

        if id1 in circuits_dict and id2 in circuits_dict:
            pairs.append((circuits_dict[id1], circuits_dict[id2]))
            labels.append(label)

    return pairs, labels

pairs, labels = generate_pair_circuits(full_df)

# K-FOLD
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(pairs)):
    print(f"\n===== Fold {fold+1}/{K} =====")

    train_pairs = [pairs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    test_pairs = [pairs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_loader = DataLoader(
        PairDataset(train_pairs, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    train_loader.batches_per_epoch = int(np.ceil(len(train_labels) / BATCH_SIZE))

    all_pairs = train_pairs + test_pairs
    a, b = zip(*all_pairs)

    model = CosineSimilarityModel.from_diagrams(a + b)
    model.initialise_weights()

    trainer = PytorchTrainer(
        model=model,
        loss_function=contrastive_similarity_loss,
        optimizer=torch.optim.Adam,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions={"acc": accuracy},
        verbose='text',
        seed=SEED
    )

    trainer.fit(train_loader)

    model.eval()
    with torch.no_grad():
        preds = model(test_pairs)
        labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

    test_acc = accuracy(preds, labels_tensor)
    train_acc = trainer.train_eval_results["acc"][-1]

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Test  Acc: {test_acc:.4f}")

    results.append((train_acc, test_acc))

# FINAL
train_scores = [r[0] for r in results]
test_scores  = [r[1] for r in results]

print("\n===== FINAL RESULTS =====")
print(f"Mean Train: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
print(f"Mean Test : {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
