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


# A class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

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





# Definition of the branch & bound algorithm.
def branch_and_bound_simple(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # Create stack using deque() structure
    stack = deque()
    counter = count()
    
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0
    lower_bound = -np.inf

    # Upper bound of the problems 
    upper_bound = np.inf

    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # Create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # ===============  Root node  ==========================

    if DEBUG_MODE:
        debug_print()
    
    # Solve relaxed problem
    model.optimize()
   
    

     # Check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth



    # Get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())
    
    # Get the objective value
    x_obj = model.ObjVal

    # Check if all variables have integer values (from the ones that are supposed to be integers)
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break

    # Found feasible solution.
    if vars_have_integer_vals:
        # If we have feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found
    
    # Otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj    
        else:
            lower_bound = x_obj


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
    stack.append(right_child)
    stack.append(left_child)
    # Solving sub problems
    # While the stack has nodes, continue solving
    while(len(stack) != 0):
        #print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # Increment total nodes by 1
        nodes += 1
        # 🔥 FIX #1: Update lower bound from best node in queue (for minimization)
        #if not isMax and len(stack) > 0:
        #    lower_bound = stack[0][0]
        
        # Get the child node on top of stack
        current_node = stack[-1]
        stack.pop()
        # Increase the nodes visited for current depth
        if current_node.depth >= len(nodes_per_depth):
            nodes_per_depth = np.pad(
                nodes_per_depth,
                (0, current_node.depth - len(nodes_per_depth) + 1),
                mode="constant",
            )

        nodes_per_depth[current_node.depth] += 1

        # Warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        #print(f"LB: {current_node.lb}")
        #print(f"UB: {current_node.ub}")

        # Update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()    
        
        if DEBUG_MODE:
            debug_print()


        # Optimize the model
        model.optimize()
        
                    # select candidate with max logit
        

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf


        else:
            # Get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # Get the objective value
            x_obj = model.ObjVal



        # If infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Check if all variables have integer values (from the ones that are supposed to be integers)
        
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break
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
            
            if x_obj > upper_bound:
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        
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
        stack.append(right_child)
        stack.append(left_child)
    
    best_solution_value= np.inf

    for i in range(0,len(solutions)):
        #print(f"{solutions[i][1]}\n")
        if solutions[i][1] < best_solution_value:
            #print("fuck this shit")
            best_sol_idx=i
            best_solution_value = solutions[i][1]


    return nodes,solutions, best_sol_idx, solutions_found
            

if __name__ == "__main__":

    print("************************    Initializing structures...    ************************")

    # --------------------------
    # Generate VRP instance
    # --------------------------
#n = 8, 9, 10, 11
#capacity_ratio = 0.30
#coord_type = uniform

    instance = generate_cvrp_instance(
                    n_customers =19,
                    seed=42,
                    Q_ratio = 0.30,
                    
                )   

    model, integer_var = build_cvrp_relaxation(instance)

    num_vars = len(model.getVars())

    vars_list = model.getVars()

    lb = np.array([v.LB for v in vars_list])
    ub = np.array([v.UB for v in vars_list])

    isMax = False  # CVRP is minimization

    # --------------------------
    # Initialize B&B structures
    # --------------------------
    if isMax:
        best_bound_per_depth = np.array([-np.inf] * num_vars)
    else:
        best_bound_per_depth = np.array([np.inf] * num_vars)

    nodes_per_depth = np.zeros(num_vars)

    # --------------------------
    # Solve
    # --------------------------
    print("************************    Solving problem...    ************************")

    start = time.time()

    nodes,solutions, best_sol_idx, solutions_found = branch_and_bound_simple(
        model,
        ub,
        lb,
        integer_var,
        best_bound_per_depth,
        nodes_per_depth
    )

    end = time.time()

    # --------------------------
    # Results
    # --------------------------
    print("========= Optimal Solution =========")

    best_solution = solutions[best_sol_idx]

    print("Decision vector:")
    print(best_solution[0])

    print(f"Objective Value: {best_solution[1]}")
    print(f"Tree depth: {best_solution[2]}")
    print(f"Time Elapsed: {end - start:.4f} seconds")
    print(f"Solutions found: {solutions_found}")
    print(f"Nodes:{nodes}")

    
    
