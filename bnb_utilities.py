import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque

import torch
import math
from typing import Tuple,Dict 
import scipy
import heapq
from itertools import count

from itertools import combinations
import math






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

        
        self.lp_obj = lp_obj
        self.x_candidate = x_candidate
        self.status = status



def is_nearly_integer(value, tolerance=1e-7):
    return abs(value - round(value)) <= tolerance



def add_static_capacity_cuts(model, instance, max_subset_size=2):
    

    n = instance["n"]
    N = n + 1
    Q = instance["Q"]
    d = instance["demands"]

    vars_by_name = {v.VarName: v for v in model.getVars()}
    customers = list(range(1, N))

    added = 0

    for size in range(2, max_subset_size + 1):
        for S_tuple in combinations(customers, size):
            S = set(S_tuple)

            demand_S = sum(d[i] for i in S)
            rhs = math.ceil(demand_S / Q)

            if rhs <= 1:
                continue

            expr = gp.LinExpr()

            for i in S:
                for j in range(N):
                    if j not in S and i != j:
                        var = vars_by_name.get(f"x_{i}_{j}")
                        if var is not None:
                            expr += var

            model.addConstr(
                expr >= rhs,
                name=f"static_cap_cut_{size}_{'_'.join(map(str, S_tuple))}",
            )

            added += 1

    model.update()
    print(f"Added {added} static root capacity cuts.")
    return added




def add_violated_capacity_cuts(
    model,
    instance,
    max_subset_size=3,
    violation_tol=1e-6,
    max_cuts=500,
):
    

    if model.status != GRB.OPTIMAL:
        return 0

    n = instance["n"]
    N = n + 1
    Q = instance["Q"]
    d = instance["demands"]

    vars_by_name = {v.VarName: v for v in model.getVars()}
    customers = list(range(1, N))

    added = 0

    for size in range(2, max_subset_size + 1):
        for S_tuple in combinations(customers, size):
            S = set(S_tuple)

            demand_S = sum(d[i] for i in S)
            rhs = math.ceil(demand_S / Q)

            if rhs <= 1:
                continue

            expr = gp.LinExpr()
            lhs_value = 0.0

            for i in S:
                for j in range(N):
                    if j not in S and i != j:
                        var = vars_by_name.get(f"x_{i}_{j}")
                        if var is not None:
                            expr += var
                            lhs_value += var.X

            if lhs_value + violation_tol < rhs:
                model.addConstr(
                    expr >= rhs,
                    name=f"violated_cap_cut_{size}_{'_'.join(map(str, S_tuple))}",
                )

                added += 1

                if added >= max_cuts:
                    model.update()
                    print(f"Added {added} violated root capacity cuts.")
                    return added

    model.update()
    print(f"Added {added} violated root capacity cuts.")
    return added

def separate_root_capacity_cuts(
    model,
    instance,
    max_subset_size=3,
    max_rounds=3,
    max_cuts_per_round=500,
    violation_tol=1e-6,
):
    

    total_added = 0

    for round_id in range(max_rounds):
        
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(
                f"Root cut round {round_id + 1}: "
                f"LP status={model.status}, stopping cut separation.",
                flush=True,
            )
            break

        root_obj_before = model.ObjVal

        
        added = add_violated_capacity_cuts(
            model=model,
            instance=instance,
            max_subset_size=max_subset_size,
            violation_tol=violation_tol,
            max_cuts=max_cuts_per_round,
        )

        total_added += added

        # If cuts were added, the current solution is invalid.
        # Re-solve before reading ObjVal.
        if added > 0:
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print(
                    f"After adding cuts, LP status={model.status}. "
                    f"Stopping cut separation.",
                    flush=True,
                )
                break

            root_obj_after = model.ObjVal
        else:
            root_obj_after = root_obj_before

        print(
            f"Root cut round {round_id + 1}/{max_rounds}: "
            f"added={added}, "
            f"root_obj_before={root_obj_before:.6f}, "
            f"root_obj_after={root_obj_after:.6f}",
            flush=True,
        )

        if added == 0:
            break

    
    if model.status != GRB.OPTIMAL:
        model.optimize()

    print(f"Total violated root cuts added: {total_added}", flush=True)

    return total_added

def strong_branching_scores(model, integer_var, x_candidate, parent_obj, max_candidates=10):

    vars_list = model.getVars()

   
    fractional = [
        i for i, is_int in enumerate(integer_var)
        if is_int and not is_nearly_integer(x_candidate[i])
    ]

    
    fractional = sorted(
        fractional,
        key=lambda i: abs(x_candidate[i] - 0.5)
    )[:max_candidates]

    candidates = fractional
    scores = []

    for idx in candidates:

     
        mL = model.copy()
        vL = mL.getVars()[idx]
        mL.setAttr("UB", vL, math.floor(x_candidate[idx]))
        mL.optimize()

        zL = mL.ObjVal if mL.status == GRB.OPTIMAL else float("inf")

        mR = model.copy()
        vR = mR.getVars()[idx]
        mR.setAttr("LB", vR, math.ceil(x_candidate[idx]))
        mR.optimize()

        zR = mR.ObjVal if mR.status == GRB.OPTIMAL else float("inf")

        delta_left = max(0, zL - parent_obj)
        delta_right = max(0, zR - parent_obj)

        scores.append(min(delta_left, delta_right))

    return np.array(candidates), np.array(scores)










def calc_route_dist(instance, route):
    dist = instance["dist"]
    total_dist = 0
    
    for i in range(len(route) - 1):
        total_dist += dist[route[i], route[i+1]]
    return total_dist

def calc_route_demand(instance, route):
    demands = instance["demands"]
    
    return sum(demands[node] for node in route)

def find_route(routes,customer):
    for route in routes:
        if customer in route:
            return route
    
def apply_2opt(instance, route):
    dist = instance["dist"]
    best_route = route.copy()
    improved = True
    
    while improved:
        improved = False
       
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                
                old_dist = dist[best_route[i-1], best_route[i]] + dist[best_route[j], best_route[j+1]]
                new_dist = dist[best_route[i-1], best_route[j]] + dist[best_route[i], best_route[j+1]]
                
                if new_dist < old_dist:
                    
                    best_route[i:j+1] = best_route[i:j+1][::-1]
                    improved = True
        if not improved:
            break
    return best_route


def Clarke_Wright(instance, skip_prob=0.5, return_solution=False):
    n = instance["n"]
    N = n + 1
    Q = instance["Q"]
    dist = instance["dist"]

    routes = [[0, i, 0] for i in range(1, n + 1)]

    savings = []

    
    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                s = dist[0, i] + dist[0, j] - dist[i, j]
                savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    for s, customer1, customer2 in savings:
        if np.random.rand() < skip_prob:
            continue

        route1 = find_route(routes, customer1)
        route2 = find_route(routes, customer2)

        if route1 is None or route2 is None or route1 == route2:
            continue

        new_route = None

        
        if customer1 == route1[-2] and customer2 == route2[1]:
            new_route = route1[:-1] + route2[1:]

        
        elif customer2 == route2[-2] and customer1 == route1[1]:
            new_route = route2[:-1] + route1[1:]

        
        elif customer1 == route1[-2] and customer2 == route2[-2]:
            reversed_route1 = route1[::-1]
            new_route = route2[:-1] + reversed_route1[1:]

       
        elif customer1 == route1[1] and customer2 == route2[1]:
            reversed_route1 = route1[::-1]
            new_route = reversed_route1[:-1] + route2[1:]

        if new_route is not None:
            if calc_route_demand(instance, new_route) <= Q:
                routes.remove(route1)
                routes.remove(route2)
                routes.append(new_route)

    optimized_routes = []
    heuristic_upper_bound = 0.0

    for route in routes:
        optimized_route = apply_2opt(instance, route)
        optimized_routes.append(optimized_route)
        heuristic_upper_bound += calc_route_dist(instance, optimized_route)

    if return_solution:
        return heuristic_upper_bound, optimized_routes

    return heuristic_upper_bound
    


def normalize_route(route):
    r = list(route)

    if len(r) == 0:
        return [0, 0]

    if r[0] != 0:
        r = [0] + r

    if r[-1] != 0:
        r = r + [0]

    return r


def compute_routes_cost(instance, routes):
    dist = instance["dist"]

    if routes is None:
        return np.inf

    total_cost = 0.0

    for route in routes:
        r = normalize_route(route)

        for i in range(len(r) - 1):
            total_cost += dist[r[i], r[i + 1]]

    return float(total_cost)


def validate_cvrp_routes(
    instance,
    routes,
    require_exact_vehicle_count=False,
    require_at_most_vehicle_count=True,
    verbose=False,
):
    n = instance["n"]
    Q = instance["Q"]
    demands = instance["demands"]
    K = get_vehicle_limit(instance)

    if routes is None:
        if verbose:
            print("Invalid heuristic solution: routes is None")
        return False

    normalized_routes = []

    for route_idx, route in enumerate(routes):
        r = normalize_route(route)

        if r[0] != 0 or r[-1] != 0:
            if verbose:
                print(f"Invalid route {route_idx}: does not start/end at depot: {route}")
            return False

        if any(v == 0 for v in r[1:-1]):
            if verbose:
                print(f"Invalid route {route_idx}: depot appears inside route: {route}")
            return False

        customers = [v for v in r if v != 0]

        if len(customers) == 0:
            continue

        normalized_routes.append(r)

    num_routes = len(normalized_routes)

    if require_exact_vehicle_count:
        if num_routes != K:
            if verbose:
                print(
                    f"Invalid number of routes: got {num_routes}, "
                    f"expected exactly K={K}"
                )
            return False

    if require_at_most_vehicle_count:
        if num_routes > K:
            if verbose:
                print(
                    f"Invalid number of routes: got {num_routes}, "
                    f"maximum allowed K={K}"
                )
            return False

    seen_customers = []

    for route_idx, route in enumerate(normalized_routes):
        customers = [v for v in route if v != 0]

        for v in customers:
            if v < 1 or v > n:
                if verbose:
                    print(f"Invalid customer {v} in route {route_idx}: {route}")
                return False

        route_demand = sum(demands[v] for v in customers)

        if route_demand > Q:
            if verbose:
                print(
                    f"Invalid route {route_idx}: demand {route_demand} > Q {Q}, "
                    f"route={route}"
                )
            return False

        seen_customers.extend(customers)

    expected = list(range(1, n + 1))

    if sorted(seen_customers) != expected:
        if verbose:
            missing = sorted(set(expected) - set(seen_customers))
            duplicates = sorted(
                [v for v in set(seen_customers) if seen_customers.count(v) > 1]
            )
            extra = sorted(set(seen_customers) - set(expected))

            print("Invalid customer coverage")
            print(f"Missing: {missing}")
            print(f"Duplicates: {duplicates}")
            print(f"Extra: {extra}")
            print(f"Seen: {sorted(seen_customers)}")
            print(f"Expected: {expected}")

        return False

    return True


def get_vehicle_limit(instance):
    if "K" in instance:
        return int(instance["K"])

    total_demand = sum(instance["demands"][1:])
    return int(math.ceil(total_demand / instance["Q"]))

def get_validated_clarke_wright_incumbent(
    instance,
    trials=1000,
    skip_prob=0.2,
    verbose=False,
):
    valid_solutions = []
    invalid_count = 0

    for _ in range(trials):
        sol_obj, routes = Clarke_Wright(
            instance,
            skip_prob=skip_prob,
            return_solution=True,
        )

        if not np.isfinite(sol_obj):
            continue

        if not validate_cvrp_routes(instance, routes, verbose=False):
            invalid_count += 1
            continue

        checked_obj = compute_routes_cost(instance, routes)

        if not np.isfinite(checked_obj):
            invalid_count += 1
            continue

        sol_obj = checked_obj

        valid_solutions.append((sol_obj, routes))

    valid_solutions.sort(key=lambda x: x[0])

    if verbose:
        print(f"Valid Clarke-Wright solutions: {len(valid_solutions)}")
        print(f"Invalid Clarke-Wright solutions rejected: {invalid_count}")

    if len(valid_solutions) == 0:
        return np.inf, None

    return valid_solutions[0]
    

