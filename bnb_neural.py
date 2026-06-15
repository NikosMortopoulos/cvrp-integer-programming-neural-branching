import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque

import torch
import math
from torch_geometric.data import Data
from typing import Tuple,Dict 
import scipy
import heapq
from itertools import count
from b_and_b import branch_and_bound_simple
from model import GNNPolicy
from gurobi_utilities import (
    extract_gurobi_state,
    strong_branching_scores_gurobi,
)
from neural_branch_utilities import *
from bnb_utilities import *


import os

os.environ.pop("GRB_LICENSE_FILE", None)


# Set sense (min/max)
isMax = True


MAX_NODES = 400_000
TIME_LIMIT_BNB = 120.0
GAP_LIMIT = 1e-4
WARM_START = True
DEBUG_MODE = True

# Total number of nodes visited
nodes = 0
dataset = []
# Lower bound of the problem
lower_bound = -np.inf

# Upper bound of the problems 
upper_bound = np.inf



class Node:
    def __init__(
        self,
        ub,
        lb,
        depth,
        vbasis,
        cbasis,
        branching_var,
        label="",
        lp_obj=None,
        x_candidate=None,
        status=None,
    ):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

        # Solved LP information for eager best-bound B&B
        self.lp_obj = lp_obj
        self.x_candidate = x_candidate
        self.status = status



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
    """
    Maps (i,j) → flat index in model.getVars()
    """
    mapping = {}
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                mapping[(i, j)] = idx
                idx += 1
    return mapping

def build_integer_mask(N, num_u_vars):
    # x-vars first, then u-vars
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





# A simple function to print debugging info
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






def branch_and_bound(
    instance,
    model,
    ub,
    lb,
    integer_var,
    best_bound_per_depth,
    nodes_per_depth,
    vbasis=[],
    cbasis=[],
    depth=0,
    optimized=False,
    policy=None,
    device=None,
    gnn_sb_top_k=10,
    branching_mode="first",
    xgb_ranker=None,
    max_nodes=100_000,
    time_limit=120.0,
    gap_limit=1e-4,
):
    
    global nodes, lower_bound, upper_bound

    # Create stack using deque() structure
    stack = []
    sb_prob = 0.1
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0
    counter = count()
    termination_status = "unknown"

    initial_solutions = []

    num_heuristic_trials = 1000 if optimized else 0

    if optimized:
        best_heuristic, best_heuristic_routes = get_validated_clarke_wright_incumbent(
            instance=instance,
            trials=10000,
            skip_prob=0.0,
            verbose=True,
        )
    else:
        best_heuristic = np.inf
        best_heuristic_routes = None




    if isMax:
        best_sol_obj = -np.inf
    else:
        if optimized:
            best_sol_obj = best_heuristic
            upper_bound = best_sol_obj

            if best_heuristic_routes is not None:
               
                solutions.append([None, best_heuristic, -1, best_heuristic_routes])
                solutions_found += 1
                best_sol_idx = 0

        else:
            best_sol_obj = np.inf
            upper_bound = np.inf

    print(f"upper_bound:{best_sol_obj}")
    # Create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # ===============  Root node  ==========================

    if DEBUG_MODE:
        debug_print()
    

    if optimized:
        

        # Solve root, add violated cuts, re-solve.
        separate_root_capacity_cuts(
            model=model,
            instance=instance,
            max_subset_size=7,
            max_rounds=7,
            max_cuts_per_round=500,
            violation_tol=1e-6,
        )
    else:
        model.optimize()

    model_relax = model.copy()

   
    

     # Check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return nodes, solutions, best_sol_idx, solutions_found, "infeasible", len(stack)
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return nodes, solutions, best_sol_idx, solutions_found, "infeasible", len(stack)



    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())
    
    # Get the objective value
    x_obj = model.ObjVal

    # Check if all variables have integer values (from the ones that are supposed to be integers)
    # If not, then select the first variable with a fractional value to be the one fixed
    fractional_vars = [
    i for i, is_int in enumerate(integer_var)
    if is_int and not is_nearly_integer(x_candidate[i])
    ]       
    vars_have_integer_vals = (len(fractional_vars) == 0)


    # Found feasible solution.
    if vars_have_integer_vals:
        # If we have feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return 0, solutions, best_sol_idx, solutions_found, "optimal", 0
    
    # Otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj    
        else:
            lower_bound = x_obj

   

    # 1. Extract graph features from current LP
        use_sb = True
        
        if not optimized:
            use_sb=False

        if branching_mode in {"full_sb", "hybrid"}:
            cand_indices, cand_scores = strong_branching_scores(
                model_relax,
                integer_var,
                x_candidate,
                x_obj,
                max_candidates=10,   # root only for now
            )
            selected_var_idx = int(cand_indices[np.argmax(cand_scores)])
        else:
            selected_var_idx = int(fractional_vars[0])


    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    
    # Warm start simplex
    if WARM_START:
        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

    # Create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)
            


    # Create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Create child nodes
    left_child = Node(left_ub, left_lb,root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Add child nodes in stack
    #stack.append(right_child)
    #stack.append(left_child)
    for child in [left_child, right_child]:
        model.setAttr("LB", model.getVars(), child.lb)
        model.setAttr("UB", model.getVars(), child.ub)

        if WARM_START and len(child.vbasis) != 0 and len(child.cbasis) != 0:
            try:
                model.setAttr("VBasis", model.getVars(), child.vbasis)
                model.setAttr("CBasis", model.getConstrs(), child.cbasis)
            except Exception:
                pass

        model.update()
        model.optimize()

        if model.status != GRB.OPTIMAL:
            continue

        child_obj = model.ObjVal

        if isMax:
            if child_obj <= lower_bound + 1e-9:
                continue
            priority = -child_obj
        else:
            if child_obj >= upper_bound - 1e-9:
                continue
            priority = child_obj

        child.lp_obj = child_obj
        child.x_candidate = np.asarray(
            model.getAttr("X", model.getVars()),
            dtype=np.float64,
        )
        child.status = model.status

        if WARM_START:
            try:
                child.vbasis = model.getAttr("VBasis", model.getVars())
                child.cbasis = model.getAttr("CBasis", model.getConstrs())
            except Exception:
                pass

        heapq.heappush(stack, (priority, next(counter), child))

    if not isMax and len(stack) > 0:
        lower_bound = stack[0][0]
    elif isMax and len(stack) > 0:
        upper_bound = -stack[0][0]
    # Solving sub problems
    # While the stack has nodes, continue solving
    while len(stack) != 0:

        if nodes >= max_nodes:
            termination_status = "node_limit"
            break

       
        if not isMax and np.isfinite(upper_bound):
            best_open_bound = stack[0][0]

            if best_open_bound >= upper_bound - 1e-9:
                lower_bound = upper_bound
                termination_status = "optimal"
                break

        # Same idea for maximization
        if isMax and np.isfinite(lower_bound):
            best_open_bound = -stack[0][0]

            if best_open_bound <= lower_bound + 1e-9:
                upper_bound = lower_bound
                termination_status = "optimal"
                break

        _, _, current_node = heapq.heappop(stack)

        nodes += 1


        if nodes >= max_nodes:
            termination_status = "node_limit"
            break

        if current_node.depth < len(nodes_per_depth):
            nodes_per_depth[current_node.depth] += 1

        if current_node.status != GRB.OPTIMAL:
            continue

        x_candidate = current_node.x_candidate
        x_obj = current_node.lp_obj

        if not isMax:
            if len(stack) > 0:
                lower_bound = stack[0][0]
            else:
                lower_bound = x_obj
        else:
            if len(stack) > 0:
                upper_bound = -stack[0][0]

        
        if not isMax and np.isfinite(upper_bound):
            gap = (upper_bound - lower_bound) / max(abs(upper_bound), 1e-9)
            gap_msg = f"{gap:.4%}"
        else:
            gap_msg = "inf"

        print(
            f"nodes={nodes} | open={len(stack)} | "
            f"LB={lower_bound:.6f} | UB={upper_bound:.6f} | gap={gap_msg}",
            flush=True,
        )



        

        # Check if all variables have integer values (from the ones that are supposed to be integers)
        
        fractional_vars = [
            i for i, is_int in enumerate(integer_var)
            if is_int and not is_nearly_integer(x_candidate[i])
        ]       
        vars_have_integer_vals = (len(fractional_vars) == 0)
        # Found feasible solution.
        # If integer solution found, then:
            # 1) - If solution improves incumbent, then store otherwise reject (optional)
            # If improves:
            # 2) - Update lb/ub for max/min respectively.
            # 3) - Check optimality condition lb=ub.
        if vars_have_integer_vals:
            if isMax:
                if lower_bound < x_obj:
                    lower_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6:  

                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1


                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        #return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1
                    
                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
               
            else:
                if upper_bound > x_obj:
                    upper_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6:  
                        
                        # Store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        #return solutions, best_sol_idx, solutions_found
                
                    # Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # Do not branch further if is an equal solution
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        
        # If lb/ub for max/min respectively, is greater/less than x_obj then prune.
        # Here we accept x_obj = lb/ub (to potentially discover another solution with equal obj value) but this is optional. 
        # If we wanted to prune, the condition is: x_obj lower-equal (<=) to lower_bound    for a maximization problem.
        # For example:
        # if isMax:
        #   if (x_obj < lower_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue
        # else:
        #   if (x_obj > upper_bound) or (abs(x_obj - lower_bound) < 1e-6):
        #       continue

        
        if isMax:
  
            if x_obj < lower_bound:
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            
            if x_obj >= upper_bound -  1e-9:
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)

        if WARM_START and len(current_node.vbasis) != 0 and len(current_node.cbasis) != 0:
            try:
                model.setAttr("VBasis", model.getVars(), current_node.vbasis)
                model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)
            except Exception:
                pass

        model.update()
        model.optimize()

        if model.status != GRB.OPTIMAL:
            continue

        x_candidate = np.asarray(
            model.getAttr("X", model.getVars()),
            dtype=np.float64,
        )
        x_obj = model.ObjVal

        fractional_vars = [
            i for i, is_int in enumerate(integer_var)
            if is_int and not is_nearly_integer(x_candidate[i])
        ]

        if len(fractional_vars) == 0:
            if not isMax and x_obj < upper_bound:
                upper_bound = x_obj
                solutions.append([x_candidate, x_obj, current_node.depth])
                solutions_found += 1
                best_sol_idx = len(solutions) - 1
            continue


        if branching_mode == "hybrid" and policy is not None:
            selected_var_idx = select_gnn_filtered_strong_branching_var(
                model=model,
                integer_var=integer_var,
                instance=instance,
                x_candidate=x_candidate,
                parent_obj=x_obj,
                fractional_vars=np.asarray(fractional_vars, dtype=np.int64),
                policy=policy,
                device=device,
                top_k=gnn_sb_top_k,
            )

            if selected_var_idx is None:
                frac_values = np.asarray(x_candidate)[fractional_vars]
                selected_var_idx = int(
                    fractional_vars[np.argmin(np.abs(frac_values - 0.5))]
                )

        if branching_mode == "gnn_top1" and policy is not None:
            top_candidates = gnn_topk_candidates(
                model=model,
                integer_var=integer_var,
                instance=instance,
                fractional_vars=np.asarray(fractional_vars, dtype=np.int64),
                policy=policy,
                device=device,
                top_k=1,
            )

            if len(top_candidates) > 0:
                selected_var_idx = int(top_candidates[0])
            else:
                frac_values = np.asarray(x_candidate)[fractional_vars]
                selected_var_idx = int(
                    fractional_vars[np.argmin(np.abs(frac_values - 0.5))]
                )

        elif branching_mode == "depth_mixed":
            selected_var_idx = select_depth_mixed_branching_var(
                model=model,
                integer_var=integer_var,
                instance=instance,
                x_candidate=x_candidate,
                parent_obj=x_obj,
                fractional_vars=np.asarray(fractional_vars, dtype=np.int64),
                current_depth=current_node.depth,
                policy=policy,
                device=device,
                xgb_ranker=xgb_ranker,
                xgb_max_candidates=50,
            )

        elif branching_mode == "hybrid" and policy is not None:
            selected_var_idx = select_gnn_filtered_strong_branching_var(
                model=model,
                integer_var=integer_var,
                instance=instance,
                x_candidate=x_candidate,
                parent_obj=x_obj,
                fractional_vars=np.asarray(fractional_vars, dtype=np.int64),
                policy=policy,
                device=device,
                top_k=gnn_sb_top_k,
            )

            if selected_var_idx is None:
                frac_values = np.asarray(x_candidate)[fractional_vars]
                selected_var_idx = int(
                    fractional_vars[np.argmin(np.abs(frac_values - 0.5))]
                )

        elif branching_mode == "xgboost" and xgb_ranker is not None:
            selected_var_idx = xgboost_select_branching_var(
                xgb_model=xgb_ranker,
                gurobi_model=model,
                integer_var=integer_var,
                instance=instance,
                x_candidate=x_candidate,
                node_depth=current_node.depth,
                max_candidates=None,
            )

            if selected_var_idx is None:
                frac_values = np.asarray(x_candidate)[fractional_vars]
                selected_var_idx = int(
                    fractional_vars[np.argmin(np.abs(frac_values - 0.5))]
                )

        elif branching_mode == "full_sb":
            candidates = np.asarray(fractional_vars, dtype=np.int64)
            scores = strong_branching_scores_gurobi(
                model=model,
                candidates=candidates,
                x_candidate=x_candidate,
                parent_obj=x_obj,
                child_time_limit=2.0,
            )
            selected_var_idx = int(candidates[np.argmax(scores)])

        elif branching_mode == "most_fractional":
            frac_values = np.asarray(x_candidate)[fractional_vars]
            selected_var_idx = int(
                fractional_vars[np.argmin(np.abs(frac_values - 0.5))]
            )

        else:
            selected_var_idx = int(fractional_vars[0])
    
        
        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")
        
        # Warm start simplex
        if WARM_START:
            # Retrieve vbasis and cbasis
            vbasis = model.getAttr("VBasis", model.getVars())
            cbasis = model.getAttr("CBasis", model.getConstrs())

        # Create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)


        # Create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        # Add child nodes in stack
        for child in [left_child, right_child]:
            model.setAttr("LB", model.getVars(), child.lb)
            model.setAttr("UB", model.getVars(), child.ub)

            if WARM_START and len(child.vbasis) != 0 and len(child.cbasis) != 0:
                try:
                    model.setAttr("VBasis", model.getVars(), child.vbasis)
                    model.setAttr("CBasis", model.getConstrs(), child.cbasis)
                except Exception:
                    pass

            model.update()
            model.optimize()

            if model.status != GRB.OPTIMAL:
                continue

            child_obj = model.ObjVal

            if isMax:
                if child_obj <= lower_bound + 1e-9:
                    continue
                priority = -child_obj
            else:
                if child_obj >= upper_bound - 1e-9:
                    continue
                priority = child_obj

            child.lp_obj = child_obj
            child.x_candidate = np.asarray(
                model.getAttr("X", model.getVars()),
                dtype=np.float64,
            )
            child.status = model.status

            if WARM_START:
                try:
                    child.vbasis = model.getAttr("VBasis", model.getVars())
                    child.cbasis = model.getAttr("CBasis", model.getConstrs())
                except Exception:
                    pass

            heapq.heappush(stack, (priority, next(counter), child))

        if not isMax and len(stack) > 0:
            lower_bound = stack[0][0]
        elif isMax and len(stack) > 0:
            upper_bound = -stack[0][0]
    
    open_nodes = len(stack)

    if termination_status == "unknown":
        if open_nodes == 0:
            termination_status = "optimal"
        else:
            termination_status = "stopped"

    if termination_status == "optimal":
        if not isMax:
            if len(stack) == 0 and np.isfinite(upper_bound):
                lower_bound = upper_bound
        else:
            if len(stack) == 0 and np.isfinite(lower_bound):
                upper_bound = lower_bound

    best_solution_value= np.inf

    for i in range(0,len(solutions)):
        #print(f"{solutions[i][1]}\n")
        if solutions[i][1] < best_solution_value:
            #print("fuck this shit")
            best_sol_idx=i
            best_solution_value = solutions[i][1]
    
    return nodes, solutions, best_sol_idx, solutions_found, termination_status, open_nodes
            

if __name__ == "__main__":

    import os
    import csv
    import json
    import pickle
    from pathlib import Path

    print("************************    Initializing structures...    ************************")

    # ============================================================
    # Global settings
    # ============================================================
    DEBUG_MODE = False

    # CVRP is minimization
    isMax = False

    # ============================================================
    # Experiment output files
    # ============================================================
    OUT_DIR = Path("experiment_logs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    PROBLEMS_DIR = OUT_DIR / "saved_problems"
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    # This TXT contains ONLY summary tables, not full logs
    LOG_FILE = OUT_DIR / "cvrp_bnb_summary_log_final_experiment.txt"

    # This CSV contains machine-readable results
    SUMMARY_CSV = OUT_DIR / "cvrp_bnb_summary.csv"

    # Set this to False if Gurobi reference makes the experiment too slow
    RUN_GUROBI_REFERENCE = True

    # Optional Gurobi time limit per generated problem
    GUROBI_TIME_LIMIT = 120

    # ============================================================
    # 10 not-too-big difficulty categories
    # 5 problems will be generated for each category
    # ============================================================
    difficulty_categories = [
        
        {
            "category_id": 7,
            "name": "C07_mid_hard_14_customers_Q025",
            "n_customers": 13,
            "Q_ratio": 0.30,
        },
        {
            "category_id": 8,
            "name": "C08_hard_15_customers_Q025",
            "n_customers": 15,
            "Q_ratio": 0.30,
        },
        { 
            "category_id": 9,
            "name": "C09_harder_16_customers_Q022",
            "n_customers": 18,
            "Q_ratio": 0.40,
        },
        {
            "category_id": 10,
            "name": "C10_hardest_18_customers_Q022",
            "n_customers": 25,
            "Q_ratio": 0.40,
        },
       
        {
            "category_id": 6,
            "name": "C06_mid_13_customers_Q025",
            "n_customers": 13,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 7,
            "name": "C07_mid_hard_14_customers_Q025",
            "n_customers": 13,
            "Q_ratio": 0.30,
        },
        {
            "category_id": 8,
            "name": "C08_hard_15_customers_Q025",
            "n_customers": 15,
            "Q_ratio": 0.30,
        },
        { 
            "category_id": 9,
            "name": "C09_harder_16_customers_Q022",
            "n_customers": 15,
            "Q_ratio": 0.35,
        },
        {
            "category_id": 10,
            "name": "C10_hardest_18_customers_Q022",
            "n_customers": 17,
            "Q_ratio": 0.35,
        },
    ]

    PROBLEMS_PER_CATEGORY = 5
    BASE_SEED = 1000

    # ============================================================
    # Save/load generated problems
    # ============================================================
    def save_problem_instance(instance, category, problem_index, seed):
        problem_name = (
            f"cat_{category['category_id']:02d}_"
            f"problem_{problem_index:02d}_"
            f"n_{category['n_customers']}_"
            f"q_{str(category['Q_ratio']).replace('.', '')}_"
            f"seed_{seed}"
        )

        pkl_path = PROBLEMS_DIR / f"{problem_name}.pkl"
        json_path = PROBLEMS_DIR / f"{problem_name}_metadata.json"

        with open(pkl_path, "wb") as f:
            pickle.dump(instance, f)

        metadata = {
            "problem_name": problem_name,
            "category_id": category["category_id"],
            "category_name": category["name"],
            "problem_index": problem_index,
            "seed": seed,
            "n_customers": category["n_customers"],
            "Q_ratio": category["Q_ratio"],
            "pkl_path": str(pkl_path),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved problem instance: {pkl_path}")
        print(f"Saved problem metadata: {json_path}")

        return str(pkl_path), str(json_path)

    def load_problem_instance(pkl_path):
        with open(pkl_path, "rb") as f:
            instance = pickle.load(f)

        return instance

    # ============================================================
    # CSV summary helper
    # ============================================================
    def append_summary_csv(rows):
        file_exists = SUMMARY_CSV.exists()

        fieldnames = [
            "category_id",
            "category_name",
            "problem_index",
            "seed",
            "n_customers",
            "Q_ratio",
            "problem_pkl_path",
            "problem_metadata_path",
            "experiment",
            "branching_mode",
            "optimized",
            "objective",
            "nodes",
            "time",
            "gap",
            "solution_type",
            "solutions_found",
            "lb",
            "ub",
            "depth",
        ]

        with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for row in rows:
                writer.writerow(row)

    # ============================================================
    # TXT summary-only helper
    # This does NOT store full logs.
    # It only appends the final comparison table per problem.
    # ============================================================
    def append_summary_txt(
        category,
        problem_index,
        seed,
        n_customers,
        Q_ratio,
        results,
        problem_pkl_path,
    ):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 120 + "\n")
            f.write("PROBLEM SUMMARY\n")
            f.write("=" * 120 + "\n")

            f.write(f"Category ID: {category['category_id']}\n")
            f.write(f"Category name: {category['name']}\n")
            f.write(f"Problem index: {problem_index}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"n_customers: {n_customers}\n")
            f.write(f"Q_ratio: {Q_ratio}\n")
            f.write(f"Saved problem: {problem_pkl_path}\n")

            f.write("\n")

            f.write(
                f"{'Experiment':60s} | "
                f"{'Obj':>12s} | "
                f"{'Nodes':>8s} | "
                f"{'Time(s)':>10s} | "
                f"{'Gap':>12s} | "
                f"{'Solution':>14s}\n"
            )
            f.write("-" * 130 + "\n")

            for r in results:
                obj_text = (
                    f"{r['objective']:.6f}"
                    if np.isfinite(r["objective"])
                    else "inf"
                )

                if np.isfinite(r["gap"]):
                    gap_text = f"{r['gap']:.4%}"
                else:
                    gap_text = "inf"

                f.write(
                    f"{r['name'][:60]:60s} | "
                    f"{obj_text:>12s} | "
                    f"{int(r['nodes']):8d} | "
                    f"{r['time']:10.4f} | "
                    f"{gap_text:>12s} | "
                    f"{r['solution_type']:>14s}\n"
                )

            f.write("-" * 130 + "\n")

    def append_error_txt(category, problem_index, seed, error):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "!" * 120 + "\n")
            f.write("ERROR WHILE RUNNING PROBLEM\n")
            f.write("!" * 120 + "\n")
            f.write(f"Category ID: {category['category_id']}\n")
            f.write(f"Category name: {category['name']}\n")
            f.write(f"Problem index: {problem_index}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Error type: {type(error).__name__}\n")
            f.write(f"Error message: {error}\n")
            f.write("!" * 120 + "\n")

    # ============================================================
    # Load GNN once
    # ============================================================
    policy, device = load_gnn_policy(
        model_path="models/cvrp_gnn/gnn_policy.pt"
    )


    XGB_MODEL_PATH = "outputs/xgboost_ranker_branching/xgboost_ranker_branching_model.joblib"

    xgb_ranker = load_xgboost_ranker(XGB_MODEL_PATH)

    # ============================================================
    # Helper function for one B&B experiment
    # ============================================================
    def run_experiment(
        instance,
        experiment_name,
        optimized,
        branching_mode,
        policy=None,
        device=None,
        xgb_ranker=None,
        gnn_sb_top_k=5,
    ):
        global nodes, lower_bound, upper_bound, isMax

        print("\n" + "=" * 100)
        print(f"RUNNING: {experiment_name}")
        print("=" * 100)

        # Reset globals
        nodes = 0
        lower_bound = -np.inf
        upper_bound = np.inf
        isMax = False

        # Fresh model for each experiment
        model, integer_var = build_cvrp_relaxation(instance)

        vars_list = model.getVars()
        num_vars = len(vars_list)

        lb = np.array([v.LB for v in vars_list])
        ub = np.array([v.UB for v in vars_list])

        if isMax:
            best_bound_per_depth = np.array([-np.inf] * num_vars)
        else:
            best_bound_per_depth = np.array([np.inf] * num_vars)

        nodes_per_depth = np.zeros(num_vars)

        start = time.time()

        nodes_out, solutions, best_sol_idx, solutions_found, termination_status, open_nodes = branch_and_bound(
            instance,
            model,
            ub,
            lb,
            integer_var,
            best_bound_per_depth,
            nodes_per_depth,
            optimized=optimized,
            policy=policy,
            device=device,
            xgb_ranker=xgb_ranker,      # <-- missing
            gnn_sb_top_k=gnn_sb_top_k,
            branching_mode=branching_mode,
            max_nodes=MAX_NODES,
        )

        end = time.time()
        elapsed = end - start

        print("\n" + "-" * 100)
        print(f"RESULT: {experiment_name}")
        print("-" * 100)

        objective = np.inf
        depth = None
        solution_type = "none"

        if len(solutions) > 0 and solutions_found > 0:
            best_solution = solutions[best_sol_idx]

            # Heuristic incumbent:
            # [None, objective, -1, routes]
            if best_solution[0] is None:
                print("Decision vector: not available for heuristic incumbent")
                print("Routes:")

                if len(best_solution) >= 4:
                    for route in best_solution[3]:
                        print(route)

                solution_type = "heuristic"

            # B&B integer incumbent:
            # [x_candidate, objective, depth]
            else:
                print("Decision vector:")
                print(best_solution[0])

                solution_type = "bnb_integer"

            objective = best_solution[1]
            depth = best_solution[2]

            print(f"Objective Value: {objective}")
            print(f"Tree depth: {depth}")

        else:
            print("No integer solution was stored in solutions.")

            if np.isfinite(upper_bound):
                print(f"Best known incumbent UB: {upper_bound}")
                objective = upper_bound
                solution_type = "upper_bound_only"
            else:
                print("No feasible incumbent available.")

        print(f"Solutions found: {solutions_found}")
        print(f"Nodes: {nodes_out}")
        print(f"Final LB: {lower_bound}")
        print(f"Final UB: {upper_bound}")
        print(f"Termination status: {termination_status}")
        print(f"Open nodes remaining: {open_nodes}")



        if np.isfinite(lower_bound) and np.isfinite(upper_bound):
            final_gap = (upper_bound - lower_bound) / max(abs(upper_bound), 1e-9)
            print(f"Final gap: {final_gap:.6%}")
        else:
            final_gap = np.inf
            print("Final gap: inf")

        print(f"Time Elapsed: {elapsed:.4f} seconds")

        return {
            "name": experiment_name,
            "branching_mode": branching_mode,
            "optimized": optimized,
            "objective": objective,
            "depth": depth,
            "solution_type": solution_type,
            "solutions_found": solutions_found,
            "nodes": nodes_out,
            "open_nodes": open_nodes,
            "termination_status": termination_status,
            "lb": lower_bound,
            "ub": upper_bound,
            "gap": final_gap,
            "time": elapsed,
        }

    # ============================================================
    # Gurobi reference solve
    # ============================================================
    def run_gurobi_reference(instance):
        print("\n" + "=" * 100)
        print("RUNNING: Gurobi MIP reference")
        print("=" * 100)

        n = instance["n"]
        N = n + 1
        Q = instance["Q"]
        dist = instance["dist"]
        d = instance["demands"]

        gurobi_model = gp.Model("CVRP_Gurobi")
        gurobi_model.Params.OutputFlag = 1
        gurobi_model.Params.TimeLimit = GUROBI_TIME_LIMIT

        x = {}
        for i in range(N):
            for j in range(N):
                if i != j:
                    x[i, j] = gurobi_model.addVar(
                        vtype=GRB.BINARY,
                        name=f"x_{i}_{j}",
                    )

        u = {}
        u[0] = gurobi_model.addVar(
            lb=0.0,
            ub=0.0,
            name="u_0",
        )

        for i in range(1, N):
            u[i] = gurobi_model.addVar(
                lb=d[i],
                ub=Q,
                name=f"u_{i}",
            )

        gurobi_model.setObjective(
            gp.quicksum(
                dist[i, j] * x[i, j]
                for i in range(N)
                for j in range(N)
                if i != j
            ),
            GRB.MINIMIZE,
        )

        for i in range(1, N):
            gurobi_model.addConstr(
                gp.quicksum(x[i, j] for j in range(N) if j != i) == 1
            )
            gurobi_model.addConstr(
                gp.quicksum(x[j, i] for j in range(N) if j != i) == 1
            )

        gurobi_model.addConstr(
            gp.quicksum(x[0, j] for j in range(1, N))
            ==
            gp.quicksum(x[i, 0] for i in range(1, N))
        )

        for i in range(1, N):
            for j in range(1, N):
                if i != j:
                    gurobi_model.addConstr(
                        u[j] >= u[i] + d[j] - Q * (1 - x[i, j])
                    )

        K = int(np.ceil(sum(d[1:]) / Q))

        gurobi_model.addConstr(
            gp.quicksum(x[0, j] for j in range(1, N)) == K
        )

        start = time.time()
        gurobi_model.optimize()
        end = time.time()

        gurobi_time = end - start

        if gurobi_model.status == GRB.OPTIMAL or gurobi_model.SolCount > 0:
            gurobi_obj = gurobi_model.ObjVal
        else:
            gurobi_obj = np.inf

        print("\n" + "=" * 100)
        print("GUROBI REFERENCE RESULT")
        print("=" * 100)

        if np.isfinite(gurobi_obj):
            print(f"Objective: {gurobi_obj:.6f}")
        else:
            print("Objective: inf")

        print(f"Time: {gurobi_time:.4f}s")
        print(f"Nodes explored: {int(gurobi_model.NodeCount)}")

        if gurobi_model.SolCount > 0:
            print(f"MIP Gap: {gurobi_model.MIPGap * 100:.6f}%")
            gurobi_gap = gurobi_model.MIPGap
        else:
            print("MIP Gap: unavailable")
            gurobi_gap = np.inf

        try:
            gurobi_bound = gurobi_model.ObjBound
        except Exception:
            gurobi_bound = np.inf

        return {
            "name": "Gurobi MIP reference",
            "branching_mode": "gurobi",
            "optimized": True,
            "objective": gurobi_obj,
            "depth": None,
            "solution_type": "gurobi",
            "solutions_found": gurobi_model.SolCount,
            "nodes": int(gurobi_model.NodeCount),
            "lb": gurobi_bound,
            "ub": gurobi_obj,
            "gap": gurobi_gap,
            "time": gurobi_time,
        }

    # ============================================================
    # Run one generated problem
    # ============================================================
    def run_problem(category, problem_index, seed):
        n_customers = category["n_customers"]
        Q_ratio = category["Q_ratio"]

        print("\n" + "#" * 120)
        print("NEW PROBLEM")
        print("#" * 120)
        print(f"Category ID: {category['category_id']}")
        print(f"Category name: {category['name']}")
        print(f"Problem index in category: {problem_index}")
        print(f"Seed: {seed}")
        print(f"n_customers: {n_customers}")
        print(f"Q_ratio: {Q_ratio}")
        print("#" * 120)

        instance = generate_cvrp_instance(
            n_customers=n_customers,
            seed=seed,
            Q_ratio=Q_ratio,
        )

        problem_pkl_path, problem_metadata_path = save_problem_instance(
            instance=instance,
            category=category,
            problem_index=problem_index,
            seed=seed,
        )

        results = []


        #results.append(
        #    run_experiment(
        #        instance=instance,
        #        experiment_name="simple B&B",
        #        optimized=False,
        #        branching_mode="most_fractional",
        #        policy=None,
        #        device=None,
        #        gnn_sb_top_k=0,
        #    )
        #)

        # ============================================================
        # 4. Optional Gurobi reference
        # ============================================================
        if RUN_GUROBI_REFERENCE:
            results.append(
                run_gurobi_reference(instance)
            )

        #results.append(
        #    run_experiment(
        #        instance=instance,
        #        experiment_name="Enhanced B&B with depth-mixed branching: SB<=2, GNN<=6, stochastic deep",
        #        optimized=True,
        #        branching_mode="depth_mixed",
        #        policy=policy,
        #        device=device,
        #        xgb_ranker=xgb_ranker,
        #        gnn_sb_top_k=0,
        #    )
        #)


        # ============================================================
        # 1. All optimizations except strong/neural branching
        # ============================================================
        #results.append(
        #    run_experiment(
        #        instance=instance,
        #        experiment_name="Enhanced B&B without SB/neural: heuristic UB + root cuts + most-fractional",
        #        optimized=True,
        #        branching_mode="most_fractional",
        #        policy=None,
        #        device=None,
        #        gnn_sb_top_k=0,
        #    )
        #)


        

        # ============================================================
        # 2. All optimizations + full strong branching at every node
        # ============================================================
        #results.append(
        #    run_experiment(
        #        instance=instance,
        #        experiment_name="Enhanced B&B with full strong branching at every node",
        #        optimized=True,
        #        branching_mode="full_sb",
        #        policy=None,
        #        device=None,
        #        gnn_sb_top_k=0,
        #    )
        #)
        results.append(
            run_experiment(
                instance=instance,
                experiment_name="Enhanced B&B with XGBoost ranker branching",
                optimized=True,
                branching_mode="xgboost",
                policy=None,
                device=None,
                xgb_ranker=xgb_ranker,
                gnn_sb_top_k=0,
            )
        )


        results.append(
            run_experiment(
                instance=instance,
                experiment_name="Enhanced B&B with pure GNN top-1 branching",
                optimized=True,
                branching_mode="gnn_top1",
                policy=policy,
                device=device,
                gnn_sb_top_k=0,
            )
        )

        # ============================================================
        # 3. All optimizations + hybrid neural strong branching
        # ============================================================
        #results.append(
        #    run_experiment(
        #        instance=instance,
        #        experiment_name="Enhanced B&B with hybrid GNN top-k + strong branching",
        #        optimized=True,
        #        branching_mode="hybrid",
        #        policy=policy,
        #        device=device,
        #        gnn_sb_top_k=2,
        #    )
        #)

        

        # ============================================================
        # Summary table for this problem
        # This prints to terminal
        # ============================================================
        print("\n" + "=" * 100)
        print("SUMMARY FOR CURRENT PROBLEM")
        print("=" * 100)

        print(
            f"{'Experiment':60s} | "
            f"{'Obj':>12s} | "
            f"{'Nodes':>8s} | "
            f"{'Open':>8s} | "
            f"{'Time(s)':>10s} | "
            f"{'Gap':>12s} | "
            f"{'Status':>12s} | "
            f"{'Solution':>14s}"

        )
        print("-" * 130)

        for r in results:
            obj_text = (
                f"{r['objective']:.6f}"
                if np.isfinite(r["objective"])
                else "inf"
            )

            if np.isfinite(r["gap"]):
                gap_text = f"{r['gap']:.4%}"
            else:
                gap_text = "inf"

            print(
                f"{r['name'][:60]:60s} | "
                f"{obj_text:>12s} | "
                f"{int(r['nodes']):8d} | "
                f"{r['time']:10.4f} | "
                f"{gap_text:>12s} | "
                f"{r['solution_type']:>14s}"
            )

        print("-" * 130)

        # ============================================================
        # Append summary only to TXT
        # ============================================================
        append_summary_txt(
            category=category,
            problem_index=problem_index,
            seed=seed,
            n_customers=n_customers,
            Q_ratio=Q_ratio,
            results=results,
            problem_pkl_path=problem_pkl_path,
        )

        # ============================================================
        # Append machine-readable CSV rows
        # ============================================================
        csv_rows = []

        for r in results:
            csv_rows.append(
                {
                    "category_id": category["category_id"],
                    "category_name": category["name"],
                    "problem_index": problem_index,
                    "seed": seed,
                    "n_customers": n_customers,
                    "Q_ratio": Q_ratio,
                    "problem_pkl_path": problem_pkl_path,
                    "problem_metadata_path": problem_metadata_path,
                    "experiment": r["name"],
                    "branching_mode": r["branching_mode"],
                    "optimized": r["optimized"],
                    "objective": r["objective"],
                    "nodes": r["nodes"],
                    "time": r["time"],
                    "gap": r["gap"],
                    "solution_type": r["solution_type"],
                    "solutions_found": r["solutions_found"],
                    "lb": r["lb"],
                    "ub": r["ub"],
                    "depth": r["depth"],
                }
            )

        append_summary_csv(csv_rows)

    # ============================================================
    # Main 10 x 5 experiment loop
    # ============================================================
    total_start = time.time()

    for category in difficulty_categories:
        for problem_index in range(1, PROBLEMS_PER_CATEGORY + 1):
            seed = BASE_SEED + category["category_id"] * 100 + problem_index

            problem_start = time.time()

            print("\n\n")
            print("*" * 120)
            print("STARTING NEW PROBLEM RUN")
            print("*" * 120)

            try:
                run_problem(
                    category=category,
                    problem_index=problem_index,
                    seed=seed,
                )

            except Exception as e:
                print("\n" + "!" * 120)
                print("ERROR WHILE RUNNING PROBLEM")
                print("!" * 120)
                print(f"Category ID: {category['category_id']}")
                print(f"Category name: {category['name']}")
                print(f"Problem index: {problem_index}")
                print(f"Seed: {seed}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("!" * 120)

                append_error_txt(
                    category=category,
                    problem_index=problem_index,
                    seed=seed,
                    error=e,
                )

            problem_end = time.time()

            print("\n" + "*" * 120)
            print("FINISHED PROBLEM RUN")
            print(f"Elapsed for this problem: {problem_end - problem_start:.4f} seconds")
            print("*" * 120)

    total_end = time.time()

    print("\n" + "=" * 120)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 120)
    print(f"Total categories: {len(difficulty_categories)}")
    print(f"Problems per category: {PROBLEMS_PER_CATEGORY}")
    print(f"Total generated problems: {len(difficulty_categories) * PROBLEMS_PER_CATEGORY}")
    print(f"Summary TXT log: {LOG_FILE}")
    print(f"CSV summary: {SUMMARY_CSV}")
    print(f"Saved problems directory: {PROBLEMS_DIR}")
    print(f"Total elapsed: {total_end - total_start:.4f} seconds")
    print("=" * 120)