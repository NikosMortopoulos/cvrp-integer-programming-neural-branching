import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque

import torch
import math
from torch_geometric.data import Data
from typing import Tuple,Dict 
import heapq
from itertools import count
from gurobi_utilities import (
    get_fractional_integer_candidates,
    extract_gurobi_state,
    strong_branching_scores_gurobi,
    make_gnn_sample,
    save_sample,
)


# Set sense (min/max)
isMax = True

WARM_START = True
DEBUG_MODE = False

# Total number of nodes visited
nodes = 0

# Lower bound of the problem
lower_bound = -np.inf

# Upper bound of the problems 
upper_bound = np.inf

def is_nearly_integer(value, tolerance=1e-7):
    return abs(value - round(value)) <= tolerance





def generate_cvrp_instance(n_customers, seed=0, Q_ratio=0.3):

    rng = np.random.default_rng(seed)

    coords = rng.uniform(0, 100, size=(n_customers+1, 2))

    demands = np.zeros(n_customers+1, dtype=int)
    demands[1:] = rng.integers(1, 10, size=n_customers)

    total_demand = demands[1:].sum()
    Q = max(1, int(Q_ratio * total_demand))

    dist = np.linalg.norm(
        coords[:, None, :] - coords[None, :, :],
        axis=-1
    )

    return {
        "coords": coords,
        "demands": demands,
        "Q": Q,
        "dist": dist,
        "n": n_customers
    }


def build_var_index_map(N):
   
    mapping = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                mapping[(i, j)] = idx
                idx += 1
    return mapping

def build_integer_mask(N, num_u_vars):
    num_x = N * (N - 1)

    integer_var = [True] * num_x + [False] * num_u_vars
    return integer_var


def build_cvrp_relaxation(instance):
    n = instance["n"]
    N = n + 1  # Total nodes including depot (node 0)
    Q = instance["Q"]
    dist = instance["dist"]
    d = instance["demands"]

    model = gp.Model("CVRP_MTZ_LP")
    model.Params.OutputFlag = 0

    # VARIABLES: Continuous relaxation for x
    x = {}
    for i in range(N):
        for j in range(N):
            if i != j:
                x[i, j] = model.addVar(lb=0.0, ub=1.0, 
                                       vtype=GRB.CONTINUOUS, 
                                       name=f"x_{i}_{j}")

    u = {}
    u[0] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name="u_0")
    for i in range(1, N):
        u[i] = model.addVar(lb=d[i], ub=Q, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # OBJECTIVE
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in range(N) for j in range(N) if i != j),
        GRB.MINIMIZE
    )

    # CONSTRAINTS
    # 1. Degree constraints (each customer visited exactly once)
    for i in range(1, N):
        model.addConstr(gp.quicksum(x[i, j] for j in range(N) if j != i) == 1)
        model.addConstr(gp.quicksum(x[j, i] for j in range(N) if j != i) == 1)

    # 2. Depot flow balance (sum out = sum in)
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, N)) == 
        gp.quicksum(x[i, 0] for i in range(1, N))
    )

    # 3. MTZ Subtour Elimination & Capacity Constraints
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                model.addConstr(u[j] >= u[i] + d[j] - Q * (1 - x[i, j]))

    K = int(np.ceil(sum(d[1:]) / Q))

    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, N)) == K
    )

    model.update()

    x_var_set = set(x.values())
    integer_var = [v in x_var_set for v in model.getVars()]

    return model,integer_var


class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

def debug_print(node:Node = None, x_obj = None, sol_status = None):
        
        print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
        print(f"UB:{upper_bound}")
        print(f"LB:{lower_bound}")
        if node is not None:
            print(f"Brancing Var: {node.branching_var}")
        if node is not None:
            print(f"Child: {node.label}")
        if node is not None:
            print(f"Depth: {node.depth}")
        if x_obj is not None:
            print(f"Simplex Objective: {x_obj}")
        if sol_status is not None:
            print(f"Solution status: {sol_status}")

        print("\n\n--------------------------------------------------\n\n")





def branch_and_bound_simple(
    model,
    ub,
    lb,
    integer_var,
    best_bound_per_depth,
    nodes_per_depth,
    instance=None,
    vbasis=[],
    cbasis=[],
    depth=0,
    collect_data=False,
    sample_out_dir="data/samples/cvrp/train",
    max_samples=100,
    max_sb_candidates=10,
    instance_seed=-1,
    category_type="normal"
):
    global nodes, lower_bound, upper_bound
    
    stack = deque()
    counter = count()
    node_record_prob = 0.05

    
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0
    lower_bound = -np.inf
    sample_counter = 0

    upper_bound = np.inf

    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # ===============  Root node  ==========================

    if DEBUG_MODE:
        debug_print()
    
    model.optimize()
   
    

    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth



    x_candidate = np.asarray(model.getAttr("X", model.getVars()), dtype=np.float64)    
    x_obj = model.ObjVal

    fractional_vars = get_fractional_integer_candidates(
        x_candidate=x_candidate,
        integer_var=integer_var,
        max_candidates=None,
    )

    vars_have_integer_vals = len(fractional_vars) == 0

    if not vars_have_integer_vals:
        if collect_data:
            sb_candidates = get_fractional_integer_candidates(
                x_candidate=x_candidate,
                integer_var=integer_var,
                max_candidates=max_sb_candidates,
            )

            state = extract_gurobi_state(
                model=model,
                integer_var=integer_var,
                instance=instance,
            )

            scores = strong_branching_scores_gurobi(
                model=model,
                candidates=sb_candidates,
                x_candidate=x_candidate,
                parent_obj=x_obj,
            )

            sample = make_gnn_sample(
                state=state,
                candidates=sb_candidates,
                candidate_scores=scores,
                node_depth=root_node.depth,
                instance_seed=instance_seed,
            )

            save_sample(sample, sample_out_dir, sample_counter,category_type,instance_seed=instance_seed)

            print(
                f"Saved root sample {sample_counter + 1}/{max_samples} "
                f"for seed={instance_seed}",
                flush=True,
            )

            sample_counter += 1
            if collect_data and sample_counter >= max_samples:
                print(
                    f"Collected {sample_counter}/{max_samples} samples at root. "
                    f"Stopping instance seed={instance_seed}.",
                    flush=True,
                )
                return nodes, solutions, best_sol_idx, solutions_found

            selected_var_idx = int(sb_candidates[np.argmax(scores)])
            
        else:
            selected_var_idx = int(fractional_vars[0])

    if vars_have_integer_vals:
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found
    
    else:
        if isMax:
            upper_bound = x_obj    
        else:
            lower_bound = x_obj


    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    
    if WARM_START:
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)
            


    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    left_child = Node(left_ub, left_lb,root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    stack.append(right_child)
    stack.append(left_child)
 
    while(len(stack) != 0):
        #print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        nodes += 1
        
        if collect_data and sample_counter >= max_samples:
            print(
                f"Already collected {sample_counter}/{max_samples} samples. "
                f"Stopping instance seed={instance_seed}.",
                flush=True,
            )
            break
        current_node = stack[-1]
        stack.pop()
        if current_node.depth >= len(nodes_per_depth):
            nodes_per_depth = np.pad(
                nodes_per_depth,
                (0, current_node.depth - len(nodes_per_depth) + 1),
                mode="constant",
            )

        nodes_per_depth[current_node.depth] += 1

        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        #print(f"LB: {current_node.lb}")
        #print(f"UB: {current_node.ub}")

        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()    
        
        if DEBUG_MODE:
            debug_print()


        model.optimize()
        
        

        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf


        else:
            x_candidate = np.asarray(model.getAttr("X", model.getVars()), dtype=np.float64)
            x_obj = model.ObjVal



        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

       
        fractional_vars = get_fractional_integer_candidates(
            x_candidate=x_candidate,
            integer_var=integer_var,
            max_candidates=None,
        )

        vars_have_integer_vals = len(fractional_vars) == 0

       
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:
                    lower_bound = x_obj
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1

                    if solutions_found == 1 or x_obj > best_sol_obj:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")

                continue

            else:
                if upper_bound > x_obj:
                    upper_bound = x_obj
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1

                    if solutions_found == 1 or x_obj < best_sol_obj:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")

                continue

   
        if isMax:
            if x_obj < lower_bound:
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            if x_obj > upper_bound:
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        should_record = (
            collect_data
            and sample_counter < max_samples
            and np.random.rand() < node_record_prob
        )

        if should_record:        
            sb_candidates = get_fractional_integer_candidates(
                x_candidate=x_candidate,
                integer_var=integer_var,
                max_candidates=max_sb_candidates,
            )

            state = extract_gurobi_state(
                model=model,
                integer_var=integer_var,
                instance=instance,
            )

            scores = strong_branching_scores_gurobi(
                model=model,
                candidates=sb_candidates,
                x_candidate=x_candidate,
                parent_obj=x_obj,
            )

            sample = make_gnn_sample(
                state=state,
                candidates=sb_candidates,
                candidate_scores=scores,
                node_depth=current_node.depth,
                instance_seed=instance_seed,
            )

            save_sample(
                sample,
                sample_out_dir,
                sample_counter,
                category_type,
                instance_seed=instance_seed,
            )

            print(
                f"Saved sample {sample_counter + 1}/{max_samples} "
                f"for seed={instance_seed} at depth={current_node.depth}",
                flush=True,
            )

            sample_counter += 1

            selected_var_idx = int(sb_candidates[np.argmax(scores)])

            if sample_counter >= max_samples:
                print(
                    f"Collected {sample_counter}/{max_samples} samples. "
                    f"Stopping instance seed={instance_seed}.",
                    flush=True,
                )
                break

        else:
            
            if len(fractional_vars) == 0:
                continue

            fractional_values = x_candidate[fractional_vars]
            selected_var_idx = int(
                fractional_vars[np.argmin(np.abs(fractional_values - 0.5))]
            )

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")
        
        if WARM_START:
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        
        stack.append(right_child)
        stack.append(left_child)
    
    if collect_data:
        print(
            f"Finished data collection for seed={instance_seed} "
            f"with {sample_counter} samples.",
            flush=True,
        )
        return nodes, solutions, best_sol_idx, solutions_found

    best_solution_value = np.inf

    for i in range(0, len(solutions)):
        if solutions[i][1] < best_solution_value:
            best_sol_idx = i
            best_solution_value = solutions[i][1]

    return nodes, solutions, best_sol_idx, solutions_found


def generate_cvrp_branching_dataset(
    num_instances=50,
    n_customers=15,
    q_ratio=0.30,
    samples_per_instance=20,
    max_sb_candidates=10,
    seed=0,
    out_dir="data/samples/cvrp/train",
    category_type="normal"
):
    global isMax, nodes, lower_bound, upper_bound

    isMax = False

    total_saved_target = num_instances * samples_per_instance

    for k in range(num_instances):
        instance_seed = seed + k

        print("=" * 80)
        print(f"Instance {k + 1}/{num_instances} | seed={instance_seed}")
        print("=" * 80)

        nodes = 0
        lower_bound = -np.inf
        upper_bound = np.inf

        instance = generate_cvrp_instance(
            n_customers=n_customers,
            seed=instance_seed,
            Q_ratio=q_ratio,
        )

        model, integer_var = build_cvrp_relaxation(instance)

        vars_list = model.getVars()
        lb = np.array([v.LB for v in vars_list], dtype=np.float64)
        ub = np.array([v.UB for v in vars_list], dtype=np.float64)

        num_vars = len(vars_list)
        best_bound_per_depth = np.array([np.inf] * num_vars)
        nodes_per_depth = np.zeros(num_vars)

        branch_and_bound_simple(
            model=model,
            ub=ub,
            lb=lb,
            integer_var=integer_var,
            best_bound_per_depth=best_bound_per_depth,
            nodes_per_depth=nodes_per_depth,
            instance=instance,
            collect_data=True,
            sample_out_dir=out_dir,
            max_samples=samples_per_instance,
            max_sb_candidates=max_sb_candidates,
            instance_seed=instance_seed,
            category_type=category_type,
        )

    print(f"Finished dataset generation. Target samples: {total_saved_target}")
    
         
if __name__ == "__main__":
    generate_cvrp_branching_dataset(
        num_instances=50,
        n_customers=80,
        q_ratio=0.25,
        samples_per_instance=20,
        max_sb_candidates=None,
        seed=0,
        out_dir="data_arc_features/samples/cvrp/debug",
        category_type="90var_30q_5rd",
    )

    generate_cvrp_branching_dataset(
        num_instances=50,
        n_customers=50,
        q_ratio=0.25,
        samples_per_instance=20,
        max_sb_candidates=None,
        seed=0,
        out_dir="data_arc_features/samples/cvrp/debug",
        category_type="90var_25q_5rd",
    )

    generate_cvrp_branching_dataset(
        num_instances=50,
        n_customers=60,
        q_ratio=0.25,
        samples_per_instance=20,
        max_sb_candidates=None,
        seed=0,
        out_dir="data_arc_features/samples/cvrp/debug",
        category_type="60var_30q_5rd",
    )


    generate_cvrp_branching_dataset(
        num_instances=50,
        n_customers=40,
        q_ratio=0.20,
        samples_per_instance=20,
        max_sb_candidates=None,
        seed=0,
        out_dir="data_arc_features/samples/cvrp/debug",
        category_type="80var_30q_5rd",
    )


    generate_cvrp_branching_dataset(
        num_instances=50,
        n_customers=50,
        q_ratio=0.24,
        samples_per_instance=20,
        max_sb_candidates=None,
        seed=0,
        out_dir="data_arc_features/samples/cvrp/debug",
        category_type="70var_30q_5rd",
    )




    