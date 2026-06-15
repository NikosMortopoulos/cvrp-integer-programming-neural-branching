import os
import csv
import json
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

import pyomo.environ as pyo


# ============================================================
# Load saved CVRP instance
# ============================================================

def load_saved_instance(pkl_path):
    with open(pkl_path, "rb") as f:
        instance = pickle.load(f)

    required_keys = ["coords", "demands", "Q", "dist", "n"]

    for key in required_keys:
        if key not in instance:
            raise ValueError(f"Missing key '{key}' in saved instance: {pkl_path}")

    return instance


def find_metadata_for_problem(pkl_path):
    pkl_path = Path(pkl_path)
    metadata_path = pkl_path.with_name(pkl_path.stem + "_metadata.json")

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


# ============================================================
# Build Pyomo CVRP MTZ model
# ============================================================

def build_pyomo_cvrp_mtz(instance):
    """
    Builds the same CVRP MTZ model as your gurobipy reference.

    Instance format:
        instance["n"]        = number of customers
        instance["Q"]        = vehicle capacity
        instance["demands"]  = array of length n+1, depot demand = 0
        instance["dist"]     = distance matrix [n+1, n+1]
    """

    n = int(instance["n"])
    N = n + 1

    Q = float(instance["Q"])
    d = np.asarray(instance["demands"], dtype=float)
    dist = np.asarray(instance["dist"], dtype=float)

    customers = list(range(1, N))
    nodes = list(range(N))
    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    K = int(np.ceil(np.sum(d[1:]) / Q))

    m = pyo.ConcreteModel(name="CVRP_MTZ_Pyomo")

    # ----------------------------
    # Sets
    # ----------------------------
    m.V = pyo.Set(initialize=nodes)
    m.C = pyo.Set(initialize=customers)
    m.A = pyo.Set(dimen=2, initialize=arcs)

    # ----------------------------
    # Parameters
    # ----------------------------
    m.Q = pyo.Param(initialize=Q)
    m.K = pyo.Param(initialize=K)

    m.demand = pyo.Param(
        m.V,
        initialize={i: float(d[i]) for i in nodes},
    )

    m.cost = pyo.Param(
        m.A,
        initialize={(i, j): float(dist[i, j]) for (i, j) in arcs},
    )

    # ----------------------------
    # Variables
    # ----------------------------
    m.x = pyo.Var(m.A, domain=pyo.Binary)

    # u[0] fixed to 0, u[i] in [d[i], Q] for customers
    def u_bounds(m, i):
        if i == 0:
            return (0.0, 0.0)
        return (float(d[i]), Q)

    m.u = pyo.Var(m.V, domain=pyo.NonNegativeReals, bounds=u_bounds)

    # ----------------------------
    # Objective
    # ----------------------------
    m.obj = pyo.Objective(
        expr=sum(m.cost[i, j] * m.x[i, j] for (i, j) in m.A),
        sense=pyo.minimize,
    )

    # ----------------------------
    # Degree constraints
    # Each customer has exactly one outgoing and one incoming arc
    # ----------------------------
    def outgoing_rule(m, i):
        return sum(m.x[i, j] for j in m.V if j != i) == 1

    m.outgoing = pyo.Constraint(m.C, rule=outgoing_rule)

    def incoming_rule(m, i):
        return sum(m.x[j, i] for j in m.V if j != i) == 1

    m.incoming = pyo.Constraint(m.C, rule=incoming_rule)

    # ----------------------------
    # Depot balance
    # ----------------------------
    m.depot_balance = pyo.Constraint(
        expr=sum(m.x[0, j] for j in m.C)
        ==
        sum(m.x[i, 0] for i in m.C)
    )

    # ----------------------------
    # Fixed number of vehicles
    # ----------------------------
    m.vehicle_count = pyo.Constraint(
        expr=sum(m.x[0, j] for j in m.C) == m.K
    )

    # ----------------------------
    # MTZ capacity/subtour constraints
    # For customer-customer arcs only
    # ----------------------------
    def mtz_rule(m, i, j):
        if i == j:
            return pyo.Constraint.Skip

        return m.u[j] >= m.u[i] + m.demand[j] - m.Q * (1 - m.x[i, j])

    m.mtz = pyo.Constraint(m.C, m.C, rule=mtz_rule)

    return m


# ============================================================
# Solve with Gurobi through Pyomo
# ============================================================

def solve_pyomo_with_gurobi(
    model,
    solver_name="gurobi",
    time_limit=120,
    mip_gap=None,
    tee=False,
    threads=None,
):
    """
    solver_name options:
        "gurobi"        -> normal Pyomo shell/persistent depending environment
        "gurobi_direct" -> direct Python interface, often better if available
    """

    solver = pyo.SolverFactory(solver_name)

    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver_name}' is not available. "
            f"Try solver_name='gurobi' or solver_name='gurobi_direct'."
        )

    # Gurobi options through Pyomo
    if time_limit is not None:
        solver.options["TimeLimit"] = int(time_limit)

    if mip_gap is not None:
        solver.options["MIPGap"] = float(mip_gap)

    if threads is not None:
        solver.options["Threads"] = int(threads)

    start = time.time()
    results = solver.solve(model, tee=tee)
    elapsed = time.time() - start

    return results, elapsed


# ============================================================
# Extract solution and stats
# ============================================================

def safe_float(value, default=np.inf):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=-1):
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def extract_solver_stats(model, results):
    termination = str(results.solver.termination_condition)
    status = str(results.solver.status)

    objective = np.inf
    bound = np.nan
    gap = np.nan

    try:
        objective = float(pyo.value(model.obj))
    except Exception:
        objective = np.inf

    # Pyomo solver result object is not always consistent across gurobi/gurobi_direct.
    # Try several safe paths.
    try:
        bound = safe_float(results.problem.lower_bound, default=np.nan)
    except Exception:
        bound = np.nan

    try:
        upper = safe_float(results.problem.upper_bound, default=np.nan)
    except Exception:
        upper = np.nan

    # For minimization:
    # lower_bound = best bound, upper_bound = incumbent objective.
    if np.isfinite(objective) and np.isfinite(bound):
        gap = (objective - bound) / max(abs(objective), 1e-9)
    elif np.isfinite(objective) and np.isfinite(upper):
        # fallback, sometimes Pyomo stores upper bound
        gap = abs(objective - upper) / max(abs(objective), 1e-9)

    nodes = -1

    try:
        nodes = safe_int(
            results.solver.statistics.branch_and_bound.number_of_created_subproblems,
            default=-1,
        )
    except Exception:
        nodes = -1

    return {
        "status": status,
        "termination": termination,
        "objective": objective,
        "best_bound": bound,
        "gap": gap,
        "nodes": nodes,
    }


def extract_routes_from_solution(model, instance, tol=0.5):
    """
    Reconstruct routes from x[i,j] = 1.
    Useful for sanity checking.
    """

    n = int(instance["n"])
    N = n + 1

    selected = {}

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            try:
                val = pyo.value(model.x[i, j])
            except Exception:
                continue

            if val is not None and val > tol:
                selected[i] = j

    routes = []

    # Routes start from depot arcs 0 -> j
    starts = []

    for j in range(1, N):
        try:
            val = pyo.value(model.x[0, j])
        except Exception:
            val = 0

        if val is not None and val > tol:
            starts.append(j)

    for start in starts:
        route = [0, start]
        current = start
        visited = set([0, start])

        while True:
            nxt = selected.get(current, None)

            if nxt is None:
                break

            route.append(nxt)

            if nxt == 0:
                break

            if nxt in visited:
                # avoid infinite loop if something is wrong
                route.append("cycle_detected")
                break

            visited.add(nxt)
            current = nxt

        routes.append(route)

    return routes


# ============================================================
# Main batch solver
# ============================================================

def solve_saved_problem_file(
    pkl_path,
    solver_name="gurobi",
    time_limit=120,
    mip_gap=None,
    tee=False,
    threads=None,
    save_routes=False,
    routes_dir=None,
):
    pkl_path = Path(pkl_path)

    instance = load_saved_instance(pkl_path)
    metadata = find_metadata_for_problem(pkl_path)

    model = build_pyomo_cvrp_mtz(instance)

    print("\n" + "=" * 100)
    print(f"Solving saved problem with Pyomo + Gurobi: {pkl_path}")
    print("=" * 100)
    print(f"n_customers = {instance['n']}")
    print(f"Q = {instance['Q']}")
    print(f"total_demand = {np.sum(instance['demands'][1:])}")
    print(f"K = {int(np.ceil(np.sum(instance['demands'][1:]) / instance['Q']))}")

    try:
        results, elapsed = solve_pyomo_with_gurobi(
            model=model,
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            tee=tee,
            threads=threads,
        )

        stats = extract_solver_stats(model, results)

        routes_path = ""

        if save_routes and np.isfinite(stats["objective"]):
            routes = extract_routes_from_solution(model, instance)

            if routes_dir is None:
                routes_dir = pkl_path.parent / "pyomo_routes"

            routes_dir = Path(routes_dir)
            routes_dir.mkdir(parents=True, exist_ok=True)

            routes_path = routes_dir / f"{pkl_path.stem}_routes.json"

            with open(routes_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "problem": str(pkl_path),
                        "objective": stats["objective"],
                        "routes": routes,
                    },
                    f,
                    indent=4,
                )

            routes_path = str(routes_path)

        row = {
            "problem_file": str(pkl_path),
            "metadata_file": str(pkl_path.with_name(pkl_path.stem + "_metadata.json")),
            "category_id": metadata.get("category_id", ""),
            "category_name": metadata.get("category_name", ""),
            "problem_index": metadata.get("problem_index", ""),
            "seed": metadata.get("seed", ""),
            "n_customers": metadata.get("n_customers", instance["n"]),
            "Q_ratio": metadata.get("Q_ratio", ""),
            "Q": instance["Q"],
            "K": int(np.ceil(np.sum(instance["demands"][1:]) / instance["Q"])),
            "solver": solver_name,
            "status": stats["status"],
            "termination": stats["termination"],
            "objective": stats["objective"],
            "best_bound": stats["best_bound"],
            "gap": stats["gap"],
            "nodes": stats["nodes"],
            "time": elapsed,
            "routes_path": routes_path,
            "error": "",
        }

        print(f"Status: {row['status']}")
        print(f"Termination: {row['termination']}")
        print(f"Objective: {row['objective']}")
        print(f"Best bound: {row['best_bound']}")
        print(f"Gap: {row['gap']}")
        print(f"Nodes: {row['nodes']}")
        print(f"Time: {row['time']:.4f}s")

        return row

    except Exception as e:
        print("ERROR:", repr(e))

        return {
            "problem_file": str(pkl_path),
            "metadata_file": str(pkl_path.with_name(pkl_path.stem + "_metadata.json")),
            "category_id": metadata.get("category_id", ""),
            "category_name": metadata.get("category_name", ""),
            "problem_index": metadata.get("problem_index", ""),
            "seed": metadata.get("seed", ""),
            "n_customers": metadata.get("n_customers", instance.get("n", "")),
            "Q_ratio": metadata.get("Q_ratio", ""),
            "Q": instance.get("Q", ""),
            "K": "",
            "solver": solver_name,
            "status": "error",
            "termination": "error",
            "objective": np.inf,
            "best_bound": np.nan,
            "gap": np.nan,
            "nodes": -1,
            "time": np.nan,
            "routes_path": "",
            "error": repr(e),
        }


def write_csv(rows, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "problem_file",
        "metadata_file",
        "category_id",
        "category_name",
        "problem_index",
        "seed",
        "n_customers",
        "Q_ratio",
        "Q",
        "K",
        "solver",
        "status",
        "termination",
        "objective",
        "best_bound",
        "gap",
        "nodes",
        "time",
        "routes_path",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\n" + "=" * 100)
    print(f"Saved CSV summary to: {csv_path}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--problems-dir",
        type=str,
        default="experiment_logs/saved_problems",
        help="Directory containing saved .pkl CVRP instances.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pkl",
        help="Glob pattern for problem files.",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="gurobi",
        choices=["gurobi", "gurobi_direct", "gurobi_persistent"],
        help="Pyomo solver interface.",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="Gurobi time limit in seconds.",
    )

    parser.add_argument(
        "--mip-gap",
        type=float,
        default=None,
        help="Optional Gurobi MIPGap, e.g. 0.0001.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional number of Gurobi threads.",
    )

    parser.add_argument(
        "--tee",
        action="store_true",
        help="Show Gurobi log output.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiment_logs/pyomo_gurobi_saved_problem_results.csv",
        help="Output CSV path.",
    )

    parser.add_argument(
        "--save-routes",
        action="store_true",
        help="Save reconstructed routes as JSON.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of problems to solve.",
    )

    args = parser.parse_args()

    problems_dir = Path(args.problems_dir)

    pkl_files = sorted(problems_dir.glob(args.pattern))

    # Exclude accidental non-problem pickle files if any
    pkl_files = [
        p for p in pkl_files
        if p.is_file()
        and not p.name.endswith("_metadata.pkl")
    ]

    if args.limit is not None:
        pkl_files = pkl_files[:args.limit]

    if len(pkl_files) == 0:
        raise RuntimeError(f"No .pkl problem files found in {problems_dir} with pattern {args.pattern}")

    print("=" * 100)
    print("PYOMO + GUROBI SOLVE SAVED CVRP PROBLEMS")
    print("=" * 100)
    print(f"Problems dir: {problems_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Number of problems: {len(pkl_files)}")
    print(f"Solver: {args.solver}")
    print(f"Time limit: {args.time_limit}")
    print(f"MIPGap: {args.mip_gap}")
    print("=" * 100)

    rows = []

    total_start = time.time()

    for idx, pkl_path in enumerate(pkl_files, start=1):
        print("\n" + "#" * 100)
        print(f"PROBLEM {idx}/{len(pkl_files)}")
        print("#" * 100)

        row = solve_saved_problem_file(
            pkl_path=pkl_path,
            solver_name=args.solver,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            tee=args.tee,
            threads=args.threads,
            save_routes=args.save_routes,
        )

        rows.append(row)

        # Write after every problem so you do not lose progress if something crashes
        write_csv(rows, args.output_csv)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 100)
    print("FINISHED ALL PYOMO + GUROBI RUNS")
    print("=" * 100)
    print(f"Problems solved/tried: {len(rows)}")
    print(f"Total time: {total_elapsed:.4f}s")
    print(f"CSV: {args.output_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()