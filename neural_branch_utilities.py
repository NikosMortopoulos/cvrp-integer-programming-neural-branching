from gurobi_utilities import *
from model import *



import joblib
import numpy as np
import torch

from gurobi_utilities import extract_gurobi_state


def load_xgboost_ranker(model_path):
    print(f"Loading XGBoost ranker from: {model_path}")
    return joblib.load(model_path)


def xgboost_select_branching_var(
    xgb_model,
    gurobi_model,
    integer_var,
    instance,
    x_candidate,
    node_depth,
    max_candidates=None,
):
    fractional_vars = get_fractional_integer_candidates(
        x_candidate=x_candidate,
        integer_var=integer_var,
        max_candidates=max_candidates,
    )

    if len(fractional_vars) == 0:
        return None

    state = extract_gurobi_state(
        model=gurobi_model,
        integer_var=integer_var,
        instance=instance,
    )

    _, _, variable_features_dict = state
    variable_features = variable_features_dict["values"]

    candidates = np.asarray(fractional_vars, dtype=np.int64)

    X = variable_features[candidates].astype(np.float32)

    # Must match training:
    # 19 variable features + node_depth + candidate_rank = 21 features
    depth_col = np.full(
        (len(candidates), 1),
        float(node_depth),
        dtype=np.float32,
    )

    rank_col = np.arange(len(candidates), dtype=np.float32).reshape(-1, 1)
    rank_col = rank_col / max(1.0, float(len(candidates) - 1))

    X = np.concatenate([X, depth_col, rank_col], axis=1)

    scores = xgb_model.predict(X)

    best_local = int(np.argmax(scores))
    branching_var = int(candidates[best_local])

    return branching_var

def gnn_topk_candidates(
    model,
    integer_var,
    instance,
    fractional_vars,
    policy,
    device,
    top_k=10,
):
    if len(fractional_vars) == 0:
        return np.asarray([], dtype=np.int64)

    state = extract_gurobi_state(
        model=model,
        integer_var=integer_var,
        instance=instance,
    )

    constraint_features_dict, edge_features_dict, variable_features_dict = state

    constraint_features = torch.as_tensor(
        constraint_features_dict["values"],
        dtype=torch.float32,
        device=device,
    )

    edge_index = torch.as_tensor(
        edge_features_dict["indices"],
        dtype=torch.long,
        device=device,
    )

    edge_attr = torch.as_tensor(
        edge_features_dict["values"],
        dtype=torch.float32,
        device=device,
    )

    variable_features = torch.as_tensor(
        variable_features_dict["values"],
        dtype=torch.float32,
        device=device,
    )

    fractional_vars_t = torch.as_tensor(
        np.asarray(fractional_vars, dtype=np.int64),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        out = policy(
            constraint_features,
            edge_index,
            edge_attr,
            variable_features,
        )

        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        candidate_logits = logits[fractional_vars_t]

        k = min(top_k, len(fractional_vars))
        top_local = torch.topk(candidate_logits, k=k, dim=0).indices

        top_candidates = fractional_vars_t[top_local].cpu().numpy()

    return top_candidates.astype(np.int64)


def select_gnn_filtered_strong_branching_var(
    model,
    integer_var,
    instance,
    x_candidate,
    parent_obj,
    fractional_vars,
    policy,
    device,
    top_k=10,
):
    """
    Hybrid brancher:
        1. GNN ranks all fractional candidates.
        2. Keep top_k candidates.
        3. Run strong branching only on those top_k.
        4. Return best strong-branching candidate.
    """

    top_candidates = gnn_topk_candidates(
        model=model,
        integer_var=integer_var,
        instance=instance,
        fractional_vars=fractional_vars,
        policy=policy,
        device=device,
        top_k=top_k,
    )

    if len(top_candidates) == 0:
        return None

    scores = strong_branching_scores_gurobi(
        model=model,
        candidates=top_candidates,
        x_candidate=np.asarray(x_candidate, dtype=np.float64),
        parent_obj=parent_obj,
        child_time_limit=2.0,
    )

    selected_var_idx = int(top_candidates[np.argmax(scores)])

    return selected_var_idx


def load_gnn_policy(
    model_path="models/cvrp_gnn/gnn_policy.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = GNNPolicy(
        var_nfeats=19,
        cons_nfeats=5,
        edge_nfeats=1,
        emb_size=32,
        n_rounds=1,
        dropout=0.05,
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "policy" in checkpoint:
        state_dict = checkpoint["policy"]
    else:
        state_dict = checkpoint

    policy.load_state_dict(state_dict)
    policy.eval()

    return policy, device


def select_depth_mixed_branching_var(
    model,
    integer_var,
    instance,
    x_candidate,
    parent_obj,
    fractional_vars,
    current_depth,
    policy=None,
    device=None,
    xgb_ranker=None,
    xgb_max_candidates=50,
):
    candidates = np.asarray(fractional_vars, dtype=np.int64)

    if len(candidates) == 0:
        return None

    # Depth 0, 1, 2: strong branching
    if current_depth <= 2:
        scores = strong_branching_scores_gurobi(
            model=model,
            candidates=candidates,
            x_candidate=np.asarray(x_candidate, dtype=np.float64),
            parent_obj=parent_obj,
            child_time_limit=2.0,
        )
        return int(candidates[np.argmax(scores)])

    # Depth 3, 4, 5, 6: GNN top-1
    if current_depth <= 6 and policy is not None:
        top_candidates = gnn_topk_candidates(
            model=model,
            integer_var=integer_var,
            instance=instance,
            fractional_vars=candidates,
            policy=policy,
            device=device,
            top_k=1,
        )

        if len(top_candidates) > 0:
            return int(top_candidates[0])

    # Depth > 6: stochastic cheap policy
    r = np.random.rand()

    # 40% XGBoost
    if r < 0.40 and xgb_ranker is not None:
        selected = xgboost_select_branching_var(
            xgb_model=xgb_ranker,
            gurobi_model=model,
            integer_var=integer_var,
            instance=instance,
            x_candidate=x_candidate,
            node_depth=current_depth,
            max_candidates=xgb_max_candidates,
        )

        if selected is not None:
            return int(selected)

    # 30% GNN
    if r < 0.70 and policy is not None:
        top_candidates = gnn_topk_candidates(
            model=model,
            integer_var=integer_var,
            instance=instance,
            fractional_vars=candidates,
            policy=policy,
            device=device,
            top_k=1,
        )

        if len(top_candidates) > 0:
            return int(top_candidates[0])

    # 30% most fractional
    frac_values = np.asarray(x_candidate)[candidates]
    return int(candidates[np.argmin(np.abs(frac_values - 0.5))])