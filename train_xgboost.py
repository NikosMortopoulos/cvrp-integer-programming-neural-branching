import os
import gzip
import pickle
import random
import json
import datetime

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib


# ============================================================
# CONFIG
# ============================================================

SAMPLES_DIR = "data_arc_features/samples/cvrp/debug"
OUT_DIR = "outputs/xgboost_ranker_branching"

TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_SEED = 42

USE_NODE_DEPTH = True
USE_CANDIDATE_RANK_FEATURE = True
USE_SCORE_NORMALIZATION = True

MODEL_NAME = "xgboost_ranker_branching_model.joblib"


# ============================================================
# LOGGING
# ============================================================

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, logfile=None):
    text = f"[{now()}] {msg}"
    print(text)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ============================================================
# LOADING
# ============================================================

def find_sample_files(samples_dir):
    files = []
    for root, _, names in os.walk(samples_dir):
        for name in names:
            if name.endswith(".pkl") or name.endswith(".pkl.gz"):
                files.append(os.path.join(root, name))
    return sorted(files)


def load_sample(path):
    # Some files are gzip-compressed even though they end with .pkl
    with open(path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# FEATURE BUILDING
# ============================================================

def sample_to_group(sample):
    """
    One sample = one B&B node = one ranking group.

    X rows = candidate variables.
    y values = relevance scores from strong branching.
    """

    state, _, best_cand, candidates, candidate_scores = sample["data"]

    _, _, variable_features_dict = state

    variable_features = variable_features_dict["values"]
    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)

    if len(candidates) == 0:
        return None

    if len(candidates) != len(candidate_scores):
        return None

    if np.any(candidates < 0) or np.any(candidates >= variable_features.shape[0]):
        return None

    X = variable_features[candidates].astype(np.float32)
    y = candidate_scores.astype(np.float32)

    # Normalize score only inside this node.
    # Convert strong branching scores to integer relevance labels.
    # Highest score gets highest relevance.
    order = np.argsort(y)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y))

    # Scale to small integer relevance values: 0, 1, 2, 3, 4
    if len(y) > 1:
        y = np.floor(4.0 * ranks / (len(y) - 1)).astype(np.int32)
    else:
        y = np.zeros_like(y, dtype=np.int32)

    extra_features = []

    if USE_NODE_DEPTH:
        depth = float(sample.get("node_depth", 0))
        depth_col = np.full((len(candidates), 1), depth, dtype=np.float32)
        extra_features.append(depth_col)

    if USE_CANDIDATE_RANK_FEATURE:
        rank = np.arange(len(candidates), dtype=np.float32).reshape(-1, 1)
        rank = rank / max(1.0, float(len(candidates) - 1))
        extra_features.append(rank)

    if extra_features:
        X = np.concatenate([X] + extra_features, axis=1)

    true_best_local = int(np.argmax(candidate_scores))

    # Dataset consistency check
    if candidates[true_best_local] != int(best_cand):
        # Use scores as source of truth
        best_local = true_best_local
    else:
        best_local = true_best_local

    return {
        "X": X,
        "y": y,
        "raw_scores": candidate_scores,
        "group_size": len(candidates),
        "best_local": best_local,
        "best_cand": int(candidates[best_local]),
    }


def build_rank_dataset(sample_files, logfile=None):
    X_list = []
    y_list = []
    group_sizes = []
    best_locals = []
    raw_scores_groups = []

    skipped = 0
    tied_groups = 0
    inconsistent_best = 0

    for path in sample_files:
        try:
            sample = load_sample(path)
            group = sample_to_group(sample)

            if group is None:
                skipped += 1
                continue

            raw_scores = group["raw_scores"]

            if np.max(raw_scores) - np.min(raw_scores) <= 1e-8:
                tied_groups += 1

            state, _, stored_best_cand, candidates, candidate_scores = sample["data"]
            candidates = np.asarray(candidates, dtype=np.int64)
            candidate_scores = np.asarray(candidate_scores, dtype=np.float32)

            score_best_cand = int(candidates[np.argmax(candidate_scores)])
            if score_best_cand != int(stored_best_cand):
                inconsistent_best += 1

            X_list.append(group["X"])
            y_list.append(group["y"])
            group_sizes.append(group["group_size"])
            best_locals.append(group["best_local"])
            raw_scores_groups.append(group["raw_scores"])

        except Exception as e:
            skipped += 1
            log(f"Skipped bad sample: {path} | error={e}", logfile)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32)
    group_sizes = np.asarray(group_sizes, dtype=np.int32)

    log(f"Loaded ranking groups: {len(group_sizes)}", logfile)
    log(f"Loaded candidate rows: {len(y)}", logfile)
    log(f"Feature dimension: {X.shape[1]}", logfile)
    log(f"Skipped samples: {skipped}", logfile)
    log(f"Tied-score groups: {tied_groups}", logfile)
    log(f"Inconsistent stored best_cand groups: {inconsistent_best}", logfile)
    log(f"Average candidates per group: {float(np.mean(group_sizes)):.2f}", logfile)

    return X, y, group_sizes, best_locals, raw_scores_groups


# ============================================================
# EVALUATION
# ============================================================

def evaluate_ranker(model, X, group_sizes, best_locals, raw_scores_groups):
    preds = model.predict(X)

    top1 = 0
    top3 = 0
    regret_sum = 0.0
    normalized_regret_sum = 0.0

    start = 0

    for group_idx, group_size in enumerate(group_sizes):
        end = start + group_size

        group_preds = preds[start:end]
        raw_scores = raw_scores_groups[group_idx]
        best_local = best_locals[group_idx]

        order = np.argsort(-group_preds)
        pred_best = int(order[0])

        if pred_best == best_local:
            top1 += 1

        if best_local in order[:3]:
            top3 += 1

        best_score = float(raw_scores[best_local])
        chosen_score = float(raw_scores[pred_best])

        regret = max(0.0, best_score - chosen_score)
        regret_sum += regret

        denom = abs(best_score) + 1e-8
        normalized_regret_sum += regret / denom

        start = end

    n = len(group_sizes)

    return {
        "top1_accuracy": top1 / max(1, n),
        "top3_accuracy": top3 / max(1, n),
        "mean_regret": regret_sum / max(1, n),
        "mean_normalized_regret": normalized_regret_sum / max(1, n),
        "num_groups": int(n),
    }


def evaluate_random_baseline(group_sizes, best_locals, raw_scores_groups, seed=42):
    rng = np.random.default_rng(seed)

    top1 = 0
    top3 = 0
    regret_sum = 0.0

    for group_idx, group_size in enumerate(group_sizes):
        raw_scores = raw_scores_groups[group_idx]
        best_local = best_locals[group_idx]

        order = rng.permutation(group_size)
        pred_best = int(order[0])

        if pred_best == best_local:
            top1 += 1

        if best_local in order[:3]:
            top3 += 1

        best_score = float(raw_scores[best_local])
        chosen_score = float(raw_scores[pred_best])
        regret_sum += max(0.0, best_score - chosen_score)

    n = len(group_sizes)

    return {
        "top1_accuracy": top1 / max(1, n),
        "top3_accuracy": top3 / max(1, n),
        "mean_regret": regret_sum / max(1, n),
        "num_groups": int(n),
    }


def evaluate_candidate_rank_baseline(group_sizes, best_locals, raw_scores_groups):
    """
    Baseline: choose first candidate in the candidate list.
    Your candidate list is usually sorted by closeness to 0.5.
    """

    top1 = 0
    top3 = 0
    regret_sum = 0.0

    for group_idx, group_size in enumerate(group_sizes):
        raw_scores = raw_scores_groups[group_idx]
        best_local = best_locals[group_idx]

        pred_best = 0

        if pred_best == best_local:
            top1 += 1

        if best_local in list(range(min(3, group_size))):
            top3 += 1

        best_score = float(raw_scores[best_local])
        chosen_score = float(raw_scores[pred_best])
        regret_sum += max(0.0, best_score - chosen_score)

    n = len(group_sizes)

    return {
        "top1_accuracy": top1 / max(1, n),
        "top3_accuracy": top3 / max(1, n),
        "mean_regret": regret_sum / max(1, n),
        "num_groups": int(n),
    }


# ============================================================
# SPLIT
# ============================================================

def split_sample_files(sample_files):
    train_files, test_files = train_test_split(
        sample_files,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train_files, val_files = train_test_split(
        train_files,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return train_files, val_files, test_files


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logfile = os.path.join(OUT_DIR, "xgboost_ranker_training_log.txt")

    log("=" * 80, logfile)
    log("XGBoost Ranker branching training started", logfile)
    log("=" * 80, logfile)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sample_files = find_sample_files(SAMPLES_DIR)

    if len(sample_files) == 0:
        raise RuntimeError(f"No sample files found in: {SAMPLES_DIR}")

    log(f"Total sample files found: {len(sample_files)}", logfile)

    train_files, val_files, test_files = split_sample_files(sample_files)

    log(f"Train sample files: {len(train_files)}", logfile)
    log(f"Val sample files:   {len(val_files)}", logfile)
    log(f"Test sample files:  {len(test_files)}", logfile)

    log("\nBuilding train ranking dataset...", logfile)
    X_train, y_train, group_train, best_train, raw_train = build_rank_dataset(train_files, logfile)

    log("\nBuilding validation ranking dataset...", logfile)
    X_val, y_val, group_val, best_val, raw_val = build_rank_dataset(val_files, logfile)

    log("\nBuilding test ranking dataset...", logfile)
    X_test, y_test, group_test, best_test, raw_test = build_rank_dataset(test_files, logfile)

    # ========================================================
    # XGBoost Ranker
    # ========================================================

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=RANDOM_SEED,
        tree_method="hist",
        eval_metric="ndcg@10",
        early_stopping_rounds=30,
    )

    log("\nTraining XGBoost Ranker...", logfile)

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[group_train, group_val],
        verbose=True,
    )

    # ========================================================
    # Evaluation
    # ========================================================

    log("\nEvaluating ranker...", logfile)

    train_rank = evaluate_ranker(model, X_train, group_train, best_train, raw_train)
    val_rank = evaluate_ranker(model, X_val, group_val, best_val, raw_val)
    test_rank = evaluate_ranker(model, X_test, group_test, best_test, raw_test)

    random_test = evaluate_random_baseline(group_test, best_test, raw_test, seed=RANDOM_SEED)
    candidate_order_test = evaluate_candidate_rank_baseline(group_test, best_test, raw_test)

    log(f"Train ranking: {train_rank}", logfile)
    log(f"Val ranking:   {val_rank}", logfile)
    log(f"Test ranking:  {test_rank}", logfile)

    log("\nBaselines on test:", logfile)
    log(f"Random baseline:         {random_test}", logfile)
    log(f"Candidate-order baseline:{candidate_order_test}", logfile)

    # ========================================================
    # Save
    # ========================================================

    model_path = os.path.join(OUT_DIR, MODEL_NAME)
    joblib.dump(model, model_path)

    results = {
        "config": {
            "samples_dir": SAMPLES_DIR,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "random_seed": RANDOM_SEED,
            "use_node_depth": USE_NODE_DEPTH,
            "use_candidate_rank_feature": USE_CANDIDATE_RANK_FEATURE,
            "use_score_normalization": USE_SCORE_NORMALIZATION,
            "model": "XGBRanker",
            "objective": "rank:pairwise",
        },
        "ranking": {
            "train": train_rank,
            "val": val_rank,
            "test": test_rank,
        },
        "baselines": {
            "random_test": random_test,
            "candidate_order_test": candidate_order_test,
        },
        "model_path": model_path,
    }

    results_path = os.path.join(OUT_DIR, "xgboost_ranker_results.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    feature_importance_path = os.path.join(OUT_DIR, "feature_importances.txt")

    with open(feature_importance_path, "w", encoding="utf-8") as f:
        importances = model.feature_importances_
        for i, imp in enumerate(importances):
            f.write(f"feature_{i}: {imp:.8f}\n")

    log("\nSaved files:", logfile)
    log(f"Model:              {model_path}", logfile)
    log(f"Results:            {results_path}", logfile)
    log(f"Feature importance: {feature_importance_path}", logfile)

    log("\nFinal test ranking:", logfile)
    log(f"Top-1 accuracy: {test_rank['top1_accuracy']:.4f}", logfile)
    log(f"Top-3 accuracy: {test_rank['top3_accuracy']:.4f}", logfile)
    log(f"Mean regret:    {test_rank['mean_regret']:.6f}", logfile)

    log("\nDone.", logfile)


if __name__ == "__main__":
    main()