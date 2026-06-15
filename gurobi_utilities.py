# gurobi_utilities.py

import os
import gzip
import pickle
import math
import datetime

import numpy as np
import torch
from torch_geometric.data import Data
from gurobipy import GRB


def log(msg, logfile=None):
    msg = f"[{datetime.datetime.now()}] {msg}"
    print(msg)
    if logfile is not None:
        with open(logfile, "a") as f:
            print(msg, file=f)


class BipartiteMILPData(Data):
    """
    PyG Data object for a bipartite MILP graph.

    constraint_features: [num_constraints, cons_nfeats]
    variable_features:   [num_variables, var_nfeats]
    edge_index:          [2, num_edges]
                         edge_index[0] = constraint index
                         edge_index[1] = variable index
    edge_attr:           [num_edges, edge_nfeats]
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([
                [self.num_constraints],
                [self.num_variables],
            ])

        if key == "candidates":
            return self.num_variables

        return super().__inc__(key, value, *args, **kwargs)


def is_nearly_integer(value, tolerance=1e-7):
    return abs(value - round(value)) <= tolerance


def get_fractional_integer_candidates(x_candidate, integer_var, max_candidates=None, eps=1e-7):
    candidates = [
        i for i, is_int in enumerate(integer_var)
        if is_int and not is_nearly_integer(x_candidate[i], eps)
    ]

    # closest to 0.5 first
    candidates = sorted(candidates, key=lambda i: abs(x_candidate[i] - 0.5))

    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    return np.asarray(candidates, dtype=np.int64)


def extract_gurobi_state(model, integer_var, instance=None, eps=1e-12):
    """
    Gurobi version of Learn2Branch utilities.extract_state().

    This extracts a bipartite MILP graph from the current solved LP relaxation.

    Returns:
        state = (
            constraint_features_dict,
            edge_features_dict,
            variable_features_dict,
        )

    Feature dimensions:
        constraint_features: [n_constraints, 5]
        edge_attr:           [n_edges, 1]
        variable_features:   [n_variables, 19]

    Variable features keep the original 19-dimensional shape, but replace
    unavailable SCIP features:
        age, inc_val, avg_inc_val

    with CVRP-specific arc features:
        arc_distance, arc_saving, demand_sum
    """

    vars_ = model.getVars()
    cons_ = model.getConstrs()

    n_vars = len(vars_)
    n_cons = len(cons_)

    # ----------------------------
    # Basic LP attributes
    # ----------------------------
    x = np.asarray(model.getAttr("X", vars_), dtype=np.float64)
    obj = np.asarray([v.Obj for v in vars_], dtype=np.float64)
    rc = np.asarray(model.getAttr("RC", vars_), dtype=np.float64)

    lb = np.asarray(
        [
            v.LB if v.LB > -GRB.INFINITY else np.nan
            for v in vars_
        ],
        dtype=np.float64,
    )

    ub = np.asarray(
        [
            v.UB if v.UB < GRB.INFINITY else np.nan
            for v in vars_
        ],
        dtype=np.float64,
    )

    has_lb = (~np.isnan(lb)).astype(np.float64)
    has_ub = (~np.isnan(ub)).astype(np.float64)

    lb_safe = np.where(np.isnan(lb), x, lb)
    ub_safe = np.where(np.isnan(ub), x + 1.0, ub)

    width = ub_safe - lb_safe
    width_safe = np.where(width > eps, width, 1.0)

    obj_norm = np.linalg.norm(obj)
    if obj_norm <= eps:
        obj_norm = 1.0

    # ----------------------------
    # Constraint matrix
    # ----------------------------
    A = model.getA()
    A_csr = A.tocsr()
    A_coo = A.tocoo()

    # ----------------------------
    # Variable features
    # ----------------------------

    # Type one-hot:
    # [binary, integer, implicit_integer, continuous]
    #
    # In your CVRP relaxation:
    # - x[i,j] variables are treated as binary branch candidates
    # - u[i] variables are continuous
    type_feats = np.zeros((n_vars, 4), dtype=np.float64)

    for i, is_int in enumerate(integer_var):
        if is_int:
            type_feats[i, 0] = 1.0   # binary
        else:
            type_feats[i, 3] = 1.0   # continuous

    coef_normalized = (obj / obj_norm).reshape(-1, 1)

    has_lb = has_lb.reshape(-1, 1)
    has_ub = has_ub.reshape(-1, 1)

    sol_is_at_lb = (np.abs(x - lb_safe) <= 1e-7).astype(np.float64).reshape(-1, 1)
    sol_is_at_ub = (np.abs(x - ub_safe) <= 1e-7).astype(np.float64).reshape(-1, 1)

    sol_frac = np.abs(x - np.round(x))
    sol_frac = sol_frac * np.asarray(integer_var, dtype=np.float64)
    sol_frac = sol_frac.reshape(-1, 1)

    # Gurobi VBasis values:
    #  0  basic
    # -1  nonbasic at lower bound
    # -2  nonbasic at upper bound
    # -3  superbasic
    #
    # We map to:
    # [lower, basic, upper, other]
    basis_feats = np.zeros((n_vars, 4), dtype=np.float64)

    try:
        vbasis = np.asarray(model.getAttr("VBasis", vars_), dtype=int)

        for i, b in enumerate(vbasis):
            if b == -1:
                basis_feats[i, 0] = 1.0
            elif b == 0:
                basis_feats[i, 1] = 1.0
            elif b == -2:
                basis_feats[i, 2] = 1.0
            else:
                basis_feats[i, 3] = 1.0

    except Exception:
        pass

    reduced_cost = (rc / obj_norm).reshape(-1, 1)

    # --------------------------------------------------
    # CVRP-specific features replacing unavailable SCIP:
    #   age, inc_val, avg_inc_val
    #
    # For x[i,j] variables:
    #   arc_distance = dist[i,j] / max_dist
    #   arc_saving   = (dist[0,i] + dist[0,j] - dist[i,j]) / (2 * max_dist)
    #   demand_sum   = (demand[i] + demand[j]) / (2 * max_demand)
    #
    # For u variables:
    #   all zero
    # --------------------------------------------------
    arc_distance = np.zeros(n_vars, dtype=np.float64)
    arc_saving = np.zeros(n_vars, dtype=np.float64)
    demand_sum = np.zeros(n_vars, dtype=np.float64)

    if instance is not None:
        n_customers = instance["n"]
        N = n_customers + 1
        dist = instance["dist"]
        demands = instance["demands"]

        max_dist = np.max(dist) + eps
        max_demand = np.max(demands) + eps

        var_idx = 0

        # Must match build_cvrp_relaxation() variable creation order:
        # for i in range(N):
        #     for j in range(N):
        #         if i != j:
        #             add x[i,j]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if var_idx >= n_vars:
                    break

                arc_distance[var_idx] = dist[i, j] / max_dist

                saving = dist[0, i] + dist[0, j] - dist[i, j]
                arc_saving[var_idx] = saving / (2.0 * max_dist)

                demand_sum[var_idx] = (
                    demands[i] + demands[j]
                ) / (2.0 * max_demand)

                var_idx += 1

    arc_distance = arc_distance.reshape(-1, 1)
    arc_saving = arc_saving.reshape(-1, 1)
    demand_sum = demand_sum.reshape(-1, 1)

    sol_val = x.reshape(-1, 1)

    variable_features = np.concatenate(
        [
            type_feats,          # 0-3
            coef_normalized,     # 4
            has_lb,              # 5
            has_ub,              # 6
            sol_is_at_lb,        # 7
            sol_is_at_ub,        # 8
            sol_frac,            # 9
            basis_feats,         # 10-13
            reduced_cost,        # 14
            arc_distance,        # 15  CVRP replaces SCIP age
            sol_val,             # 16
            arc_saving,          # 17  CVRP replaces SCIP inc_val
            demand_sum,          # 18  CVRP replaces SCIP avg_inc_val
        ],
        axis=1,
    ).astype(np.float32)

    assert variable_features.shape[1] == 19

    variable_features_dict = {
        "names": [
            "type_binary",
            "type_integer",
            "type_implicit_integer",
            "type_continuous",
            "coef_normalized",
            "has_lb",
            "has_ub",
            "sol_is_at_lb",
            "sol_is_at_ub",
            "sol_frac",
            "basis_lower",
            "basis_basic",
            "basis_upper",
            "basis_other",
            "reduced_cost",
            "arc_distance",
            "sol_val",
            "arc_saving",
            "demand_sum",
        ],
        "values": variable_features,
    }

    # ----------------------------
    # Constraint features: 5
    # ----------------------------
    Ax = A_csr @ x

    rhs = np.asarray([c.RHS for c in cons_], dtype=np.float64)
    sense = np.asarray([c.Sense for c in cons_])

    row_l2 = np.sqrt(np.asarray(A_csr.power(2).sum(axis=1)).ravel())
    row_l2_safe = np.where(row_l2 > eps, row_l2, 1.0)

    obj_cos = np.zeros(n_cons, dtype=np.float64)

    for r in range(n_cons):
        start = A_csr.indptr[r]
        end = A_csr.indptr[r + 1]

        cols = A_csr.indices[start:end]
        vals = A_csr.data[start:end]

        denom = row_l2_safe[r] * obj_norm
        if denom > eps:
            obj_cos[r] = np.dot(vals, obj[cols]) / denom

    bias = rhs / row_l2_safe

    slack = np.zeros(n_cons, dtype=np.float64)
    sense_value = np.zeros(n_cons, dtype=np.float64)

    for r in range(n_cons):
        if sense[r] == "<":
            slack[r] = rhs[r] - Ax[r]
            sense_value[r] = -1.0
        elif sense[r] == ">":
            slack[r] = Ax[r] - rhs[r]
            sense_value[r] = 1.0
        else:
            slack[r] = abs(Ax[r] - rhs[r])
            sense_value[r] = 0.0

    is_tight = (np.abs(slack) <= 1e-7).astype(np.float64)

    try:
        dual = np.asarray(model.getAttr("Pi", cons_), dtype=np.float64)
    except Exception:
        dual = np.zeros(n_cons, dtype=np.float64)

    dual_norm = np.linalg.norm(dual)
    if dual_norm <= eps:
        dual_norm = 1.0

    dual_normed = dual / dual_norm

    constraint_features = np.stack(
        [
            obj_cos,
            bias,
            is_tight,
            sense_value,
            dual_normed,
        ],
        axis=1,
    ).astype(np.float32)

    assert constraint_features.shape[1] == 5

    constraint_features_dict = {
        "names": [
            "obj_cosine_similarity",
            "bias",
            "is_tight",
            "sense_value",
            "dualsol_val_normalized",
        ],
        "values": constraint_features,
    }

    # ----------------------------
    # Edge features
    # ----------------------------
    edge_index = np.vstack([A_coo.row, A_coo.col]).astype(np.int64)

    edge_attr = (
        A_coo.data / row_l2_safe[A_coo.row]
    ).reshape(-1, 1).astype(np.float32)

    edge_features_dict = {
        "names": ["coef_normalized"],
        "indices": edge_index,
        "values": edge_attr,
    }

    # ----------------------------
    # Numerical cleanup
    # ----------------------------
    variable_features_dict["values"] = np.nan_to_num(
        variable_features_dict["values"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    constraint_features_dict["values"] = np.nan_to_num(
        constraint_features_dict["values"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    edge_features_dict["values"] = np.nan_to_num(
        edge_features_dict["values"],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # Same order as original Learn2Branch:
    # constraint_features, edge_features, variable_features
    state = (
        constraint_features_dict,
        edge_features_dict,
        variable_features_dict,
    )

    return state


def strong_branching_scores_gurobi(
    model,
    candidates,
    x_candidate,
    parent_obj,
    infeasible_score=1e6,
    child_time_limit=2.0,
):
    scores = []

    for c_id, idx in enumerate(candidates, start=1):
       #print(f"    strong branching candidate {c_id}/{len(candidates)}: var {idx}", flush=True)

        value = x_candidate[idx]
        floor_value = math.floor(value)
        ceil_value = math.ceil(value)

        # -------------------------
        # Left child: x_j <= floor
        # -------------------------
        left = model.copy()
        left.Params.OutputFlag = 0
        left.Params.TimeLimit = child_time_limit
        left.Params.Method = 1

        left_var = left.getVars()[idx]
        left_var.UB = floor_value
        left.update()

        #print(f"        left solve var {idx}", flush=True)
        left.optimize()
        #print(f"        left done var {idx}, status={left.status}", flush=True)

        if left.status == GRB.OPTIMAL:
            left_delta = max(0.0, left.ObjVal - parent_obj)
        elif left.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            left_delta = infeasible_score
        elif left.status == GRB.TIME_LIMIT:
            left_delta = 0.0
        else:
            left_delta = 0.0

        # -------------------------
        # Right child: x_j >= ceil
        # -------------------------
        right = model.copy()
        right.Params.OutputFlag = 0
        right.Params.TimeLimit = child_time_limit
        right.Params.Method = 1

        right_var = right.getVars()[idx]
        right_var.LB = ceil_value
        right.update()

        #print(f"        right solve var {idx}", flush=True)
        right.optimize()
        #print(f"        right done var {idx}, status={right.status}", flush=True)

        if right.status == GRB.OPTIMAL:
            right_delta = max(0.0, right.ObjVal - parent_obj)
        elif right.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
            right_delta = infeasible_score
        elif right.status == GRB.TIME_LIMIT:
            right_delta = 0.0
        else:
            right_delta = 0.0

        score = min(left_delta, right_delta)
        scores.append(score)

        # Important: free copied models
        left.dispose()
        right.dispose()

    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0, posinf=infeasible_score, neginf=0.0)

    return scores


def make_gnn_sample(state, candidates, candidate_scores, node_depth, instance_seed=-1):
    """
    Save format compatible with the original Learn2Branch idea:

    sample["data"] = [
        state,
        khalil_state,
        best_cand,
        cands,
        cand_scores
    ]

    Here khalil_state is None because Gurobi does not expose SCIP's Khalil state.
    """

    candidates = np.asarray(candidates, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)

    best_cand = int(candidates[np.argmax(candidate_scores)])

    return {
        "seed": instance_seed,
        "node_depth": int(node_depth),
        "data": [
            state,
            None,
            best_cand,
            candidates.tolist(),
            candidate_scores.tolist(),
        ],
    }


def save_sample(sample, out_dir, sample_id,category_type,instance_seed=None):
    os.makedirs(out_dir, exist_ok=True)

    if instance_seed is None:
        filename = os.path.join(out_dir, f"sample_2_{sample_id}.pkl")
    else:
        filename = os.path.join(out_dir, f"sample_{category_type}_seed{instance_seed}_{sample_id}.pkl")

    with gzip.open(filename, "wb") as f:
        pickle.dump(sample, f)

    return filename


def sample_to_pyg(sample):
    """
    Convert saved Learn2Branch-style dict into PyG data used by model.py.
    """

    state, khalil_state, best_cand, cands, cand_scores = sample["data"]

    constraint_features_dict, edge_features_dict, variable_features_dict = state

    variable_features = variable_features_dict["values"]
    edge_index = edge_features_dict["indices"]
    edge_attr = edge_features_dict["values"]
    constraint_features = constraint_features_dict["values"]

    cands = np.asarray(cands, dtype=np.int64)
    cand_scores = np.asarray(cand_scores, dtype=np.float32)

    best_local = int(np.where(cands == best_cand)[0][0])

    data = BipartiteMILPData()

    data.constraint_features = torch.from_numpy(constraint_features).float()
    data.edge_index = torch.from_numpy(edge_index).long()
    data.edge_attr = torch.from_numpy(edge_attr).float()
    data.variable_features = torch.from_numpy(variable_features).float()

    data.candidates = torch.from_numpy(cands).long()
    data.nb_candidates = torch.tensor([len(cands)], dtype=torch.long)
    data.candidate_scores = torch.from_numpy(cand_scores).float()
    data.candidate_choices = torch.tensor([best_local], dtype=torch.long)

    data.num_constraints = constraint_features.shape[0]
    data.num_variables = variable_features.shape[0]

    return data


class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files):
        self.sample_files = list(sample_files)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        with gzip.open(self.sample_files[idx], "rb") as f:
            sample = pickle.load(f)

        return sample_to_pyg(sample)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_size = int(pad_sizes.max().item())
    output = input_.new_full((len(pad_sizes), max_size), pad_value)

    start = 0
    for i, size in enumerate(pad_sizes):
        size = int(size.item())
        output[i, :size] = input_[start:start + size]
        start += size

    return output


