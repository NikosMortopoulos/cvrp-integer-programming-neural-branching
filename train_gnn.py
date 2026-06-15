# train_gnn_cvrp.py

import os
import glob
import csv
import json
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from model import GNNPolicy
from gurobi_utilities import GraphDataset, pad_tensor, log


# ============================================================
# Configuration
# ============================================================

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "data_arc_features/samples/cvrp/debug"
VALID_DIR = ""

MODEL_DIR = "models/cvrp_gnn"
MODEL_PATH = os.path.join(MODEL_DIR, "gnn_policy.pt")
LOG_PATH = os.path.join(MODEL_DIR, "train_log.txt")
CSV_LOG_PATH = os.path.join(MODEL_DIR, "train_log.csv")
SPLIT_PATH = os.path.join(MODEL_DIR, "data_split.json")
TEST_LOG_PATH = os.path.join(MODEL_DIR, "test_results.json")

BATCH_SIZE = 64
LR = 1e-2
MAX_EPOCHS = 20
PATIENCE = 4
TOP_K = [1, 3, 5, 10]

VALID_RATIO = 0.10
TEST_RATIO = 0.10

VAR_NFEATS = 19
CONS_NFEATS = 5
EDGE_NFEATS = 1
EMB_SIZE = 32
N_ROUNDS = 1
DROPOUT = 0.1


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Train / valid / test split
# ============================================================

def save_split(train_files, valid_files, test_files):
    split_data = {
        "train_files": train_files,
        "valid_files": valid_files,
        "test_files": test_files,
    }

    with open(SPLIT_PATH, "w") as f:
        json.dump(split_data, f, indent=2)


def load_split():
    if not os.path.exists(SPLIT_PATH):
        return None

    with open(SPLIT_PATH, "r") as f:
        return json.load(f)


def make_or_load_split(all_files):
    """
    Creates a fixed train/valid/test split.

    If data_split.json already exists, it reuses the same split.
    This is important because the test set must stay untouched and fixed.
    """

    old_split = load_split()

    if old_split is not None:
        train_files = old_split["train_files"]
        valid_files = old_split["valid_files"]
        test_files = old_split["test_files"]

        return train_files, valid_files, test_files

    files = list(all_files)
    random.shuffle(files)

    n = len(files)
    n_test = int(TEST_RATIO * n)
    n_valid = int(VALID_RATIO * n)

    test_files = files[:n_test]
    valid_files = files[n_test:n_test + n_valid]
    train_files = files[n_test + n_valid:]

    save_split(train_files, valid_files, test_files)

    return train_files, valid_files, test_files


# ============================================================
# CSV logging
# ============================================================

def init_csv_log():
    header = [
        "epoch",

        "train_loss",
        "valid_loss",

        "train_acc_at_1",
        "train_acc_at_3",
        "train_acc_at_5",
        "train_acc_at_10",

        "valid_acc_at_1",
        "valid_acc_at_3",
        "valid_acc_at_5",
        "valid_acc_at_10",

        "train_avg_candidates",
        "train_min_candidates",
        "train_max_candidates",

        "valid_avg_candidates",
        "valid_min_candidates",
        "valid_max_candidates",

        "train_rank_percentile",
        "valid_rank_percentile",

        "train_mrr",
        "valid_mrr",

        "train_base_loss",
        "valid_base_loss",
    ]

    with open(CSV_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_csv_log(
    epoch,
    train_loss,
    valid_loss,
    train_kacc,
    valid_kacc,
    train_stats,
    valid_stats,
):
    row = [
        epoch,

        train_loss,
        valid_loss,

        train_kacc[0],
        train_kacc[1],
        train_kacc[2],
        train_kacc[3],

        valid_kacc[0],
        valid_kacc[1],
        valid_kacc[2],
        valid_kacc[3],

        train_stats["avg_candidates"],
        train_stats["min_candidates"],
        train_stats["max_candidates"],

        valid_stats["avg_candidates"],
        valid_stats["min_candidates"],
        valid_stats["max_candidates"],

        train_stats["mean_rank_percentile"],
        valid_stats["mean_rank_percentile"],

        train_stats["mean_mrr"],
        valid_stats["mean_mrr"],

        train_stats["base_loss"],
        valid_stats["base_loss"],
    ]

    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# PreNorm initialization
# ============================================================

def run_prenorm_initialization(policy, train_loader):
    """
    Initializes all PreNormLayer objects using only the training set.

    This step must happen before normal training.
    It estimates shift/scale statistics for the PreNormLayer modules.
    Validation and test data are never used here.
    """

    log("Starting PreNorm initialization...", LOG_PATH)

    policy.train()
    policy.pre_train_init()

    prenorm_round = 0

    while True:
        prenorm_round += 1

        for batch in train_loader:
            batch = batch.to(DEVICE)

            policy.pre_train(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

        updated_layer = policy.pre_train_next()

        if updated_layer is None:
            break

        log(
            f"PreNorm round {prenorm_round}: initialized {updated_layer.__class__.__name__}",
            LOG_PATH,
        )

    log("Finished PreNorm initialization.", LOG_PATH)


# ============================================================
# Train / eval process
# ============================================================

def process(policy, loader, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        policy.train()
    else:
        policy.eval()

    total_loss = 0.0
    total_base_loss = 0.0

    total_graphs = 0
    total_kacc = np.zeros(len(TOP_K), dtype=np.float64)

    total_candidates = 0
    min_candidates = 10**9
    max_candidates = 0

    total_rank_percentile = 0.0
    total_mrr = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = batch.to(DEVICE)

            logits, variable_embeddings = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                return_embeddings=True,
            )

            cand_logits = logits[batch.candidates]
            cand_logits = pad_tensor(cand_logits, batch.nb_candidates)

            true_choices = batch.candidate_choices.view(-1)

            base_loss = F.cross_entropy(
                cand_logits,
                true_choices,
                reduction="mean",
            )

            loss = base_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()

            true_scores = pad_tensor(
                batch.candidate_scores,
                batch.nb_candidates,
                pad_value=-1e8,
            )

            true_best = true_scores.max(dim=-1, keepdim=True).values

            kacc = []

            for k in TOP_K:
                kk = min(k, cand_logits.shape[-1])

                pred_topk = cand_logits.topk(kk, dim=-1).indices
                pred_scores = true_scores.gather(-1, pred_topk)

                acc = (pred_scores == true_best).any(dim=-1).float().mean().item()
                kacc.append(acc)

            pred_order = torch.argsort(cand_logits, dim=-1, descending=True)

            batch_rank_percentiles = []
            batch_mrr = []

            for i in range(batch.num_graphs):
                n = int(batch.nb_candidates[i].item())
                true_i = int(true_choices[i].item())

                order_i = pred_order[i, :n]

                rank0 = (order_i == true_i).nonzero(as_tuple=False).item()
                rank1 = rank0 + 1

                batch_rank_percentiles.append(rank1 / n)
                batch_mrr.append(1.0 / rank1)

            num_graphs = batch.num_graphs

            total_loss += loss.item() * num_graphs
            total_base_loss += base_loss.item() * num_graphs

            total_kacc += np.asarray(kacc) * num_graphs
            total_graphs += num_graphs

            total_candidates += int(batch.nb_candidates.sum().item())
            min_candidates = min(
                min_candidates,
                int(batch.nb_candidates.min().item()),
            )
            max_candidates = max(
                max_candidates,
                int(batch.nb_candidates.max().item()),
            )

            total_rank_percentile += sum(batch_rank_percentiles)
            total_mrr += sum(batch_mrr)

    avg_loss = total_loss / total_graphs
    avg_base_loss = total_base_loss / total_graphs
    avg_kacc = total_kacc / total_graphs

    stats = {
        "avg_candidates": total_candidates / total_graphs,
        "min_candidates": min_candidates,
        "max_candidates": max_candidates,
        "mean_rank_percentile": total_rank_percentile / total_graphs,
        "mean_mrr": total_mrr / total_graphs,
        "base_loss": avg_base_loss,
    }

    return avg_loss, avg_kacc, stats


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(policy, epoch, valid_loss):
    checkpoint = {
        "epoch": epoch,
        "valid_loss": valid_loss,
        "policy": policy.state_dict(),
        "config": {
            "var_nfeats": VAR_NFEATS,
            "cons_nfeats": CONS_NFEATS,
            "edge_nfeats": EDGE_NFEATS,
            "emb_size": EMB_SIZE,
            "n_rounds": N_ROUNDS,
            "dropout": DROPOUT,
            "top_k": TOP_K,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "seed": SEED,
            "prenorm": True,
        },
    }

    torch.save(checkpoint, MODEL_PATH)


def evaluate_test_set(policy, test_loader):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    policy.load_state_dict(checkpoint["policy"])

    test_loss, test_kacc, test_stats = process(
        policy,
        test_loader,
        optimizer=None,
    )

    test_results = {
        "test_loss": float(test_loss),
        "test_acc_at_1": float(test_kacc[0]),
        "test_acc_at_3": float(test_kacc[1]),
        "test_acc_at_5": float(test_kacc[2]),
        "test_acc_at_10": float(test_kacc[3]),
        "test_stats": {
            "avg_candidates": float(test_stats["avg_candidates"]),
            "min_candidates": int(test_stats["min_candidates"]),
            "max_candidates": int(test_stats["max_candidates"]),
            "mean_rank_percentile": float(test_stats["mean_rank_percentile"]),
            "mean_mrr": float(test_stats["mean_mrr"]),
            "base_loss": float(test_stats["base_loss"]),
        },
    }

    with open(TEST_LOG_PATH, "w") as f:
        json.dump(test_results, f, indent=2)

    return test_results


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Fresh logs for each run
    open(LOG_PATH, "w").close()
    init_csv_log()

    all_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "sample_*.pkl")))

    if len(all_files) == 0:
        raise RuntimeError(f"No sample files found in: {TRAIN_DIR}")

    if VALID_DIR and os.path.exists(VALID_DIR):
        train_files = all_files
        valid_files = sorted(glob.glob(os.path.join(VALID_DIR, "sample_*.pkl")))

        random.shuffle(train_files)

        n_test = int(TEST_RATIO * len(train_files))
        test_files = train_files[:n_test]
        train_files = train_files[n_test:]

        save_split(train_files, valid_files, test_files)
    else:
        train_files, valid_files, test_files = make_or_load_split(all_files)

    log(f"Device: {DEVICE}", LOG_PATH)
    log(f"Total samples: {len(all_files)}", LOG_PATH)
    log(f"Train samples: {len(train_files)}", LOG_PATH)
    log(f"Valid samples: {len(valid_files)}", LOG_PATH)
    log(f"Test samples: {len(test_files)}", LOG_PATH)
    log(f"Split file: {SPLIT_PATH}", LOG_PATH)
    log(f"CSV log file: {CSV_LOG_PATH}", LOG_PATH)

    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)
    test_data = GraphDataset(test_files)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    policy = GNNPolicy(
        var_nfeats=VAR_NFEATS,
        cons_nfeats=CONS_NFEATS,
        edge_nfeats=EDGE_NFEATS,
        emb_size=EMB_SIZE,
        n_rounds=N_ROUNDS,
        dropout=DROPOUT,
    ).to(DEVICE)

    # --------------------------------------------------------
    # PreNorm initialization
    # --------------------------------------------------------
    run_prenorm_initialization(policy, train_loader)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=LR,
    )

    best_valid_loss = float("inf")
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_kacc, train_stats = process(
            policy,
            train_loader,
            optimizer=optimizer,
        )

        valid_loss, valid_kacc, valid_stats = process(
            policy,
            valid_loader,
            optimizer=None,
        )

        train_msg = " ".join(
            [
                f"train_acc@{k}: {acc:.4f}"
                for k, acc in zip(TOP_K, train_kacc)
            ]
        )

        valid_msg = " ".join(
            [
                f"valid_acc@{k}: {acc:.4f}"
                for k, acc in zip(TOP_K, valid_kacc)
            ]
        )

        log(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"valid_loss={valid_loss:.5f} | "
            f"{train_msg} | "
            f"{valid_msg} | "
            f"train_cands_avg={train_stats['avg_candidates']:.1f} "
            f"train_cands_min={train_stats['min_candidates']} "
            f"train_cands_max={train_stats['max_candidates']} | "
            f"valid_cands_avg={valid_stats['avg_candidates']:.1f} "
            f"valid_cands_min={valid_stats['min_candidates']} "
            f"valid_cands_max={valid_stats['max_candidates']} | "
            f"train_rank_pct={train_stats['mean_rank_percentile']:.4f} "
            f"valid_rank_pct={valid_stats['mean_rank_percentile']:.4f} | "
            f"train_mrr={train_stats['mean_mrr']:.4f} "
            f"valid_mrr={valid_stats['mean_mrr']:.4f} | "
            f"train_base_loss={train_stats['base_loss']:.5f} "
            f"valid_base_loss={valid_stats['base_loss']:.5f}",
            LOG_PATH,
        )

        append_csv_log(
            epoch,
            train_loss,
            valid_loss,
            train_kacc,
            valid_kacc,
            train_stats,
            valid_stats,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            bad_epochs = 0

            save_checkpoint(
                policy=policy,
                epoch=epoch,
                valid_loss=valid_loss,
            )

            log(f"Saved best model at epoch {epoch}.", LOG_PATH)
        else:
            bad_epochs += 1

        if bad_epochs >= PATIENCE:
            log("Early stopping.", LOG_PATH)
            break

    log(f"Best valid loss: {best_valid_loss:.5f}", LOG_PATH)

    test_results = evaluate_test_set(
        policy=policy,
        test_loader=test_loader,
    )

    log("Final test results:", LOG_PATH)
    log(json.dumps(test_results, indent=2), LOG_PATH)

    print("\nFinal test results")
    print(json.dumps(test_results, indent=2))


if __name__ == "__main__":
    main()