import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from typing import List, Dict, Tuple, Literal, Optional

# --- 1. Helper Functions ---

def ts(u: int, v: int) -> Tuple[int, int]:
    """Returns a sorted tuple for undirected edges."""
    return tuple(sorted((u, v)))

# --- 2. Path System Generation (Logic from previous discussion) ---

def derive_l_from_r(r: List[int], n: int) -> List[int]:
    l = [0] * n
    for i in range(n):
        reach = r[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            if step > l[target]:
                l[target] = step
    return l

def _is_legal_neighborly_vectors(r: List[int], l: List[int]) -> bool:
    n = len(r)
    if n == 0 or len(l) != n: return False
    for i in range(n):
        if r[i] < 1 or l[i] < 1: return False
        if r[i] + l[i] > n - 1: return False
        if l[(i + r[i]) % n] < r[i]: return False
        if r[(i - l[i]) % n] < l[i]: return False
        if r[(i + 1) % n] < r[i] - 1: return False
        if l[(i - 1) % n] < l[i] - 1: return False
    return True

def generate_wheel_paths(R: List[int]) -> Tuple[Dict[Tuple[int, int], Tuple[int, ...]], List[int]]:
    """
    Generates the shortest path system for a wheel graph given a CW reach vector R.
    Vertices: 0 (Center), 1..n (Rim).
    """
    n = len(R)
    # L[i] tells how far back (CCW) vertex i+1 can reach.
    L = derive_l_from_r(R, n)
    if not _is_legal_neighborly_vectors(R, L):
        raise ValueError("The given R vector is not a legal neighborly vector.")

    all_paths = {}
    vertices = [0] + list(range(1, n + 1))

    # Helpers for circular distances
    def get_cw_dist_and_path(u, v):
        if v >= u:
            return v - u, tuple(range(u, v + 1))
        else:
            return (n - u) + v, tuple(range(u, n + 1)) + tuple(range(1, v + 1))

    def get_ccw_dist_and_path(u, v):
        d, p = get_cw_dist_and_path(v, u)
        return d, p[::-1]

    # Generate paths
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            
            # Center involved
            if u == 0:
                all_paths[(u, v)] = (0, v)
                continue
            
            # Rim-Rim logic
            u_idx = u - 1
            dist_cw, path_cw = get_cw_dist_and_path(u, v)
            dist_ccw, path_ccw = get_ccw_dist_and_path(u, v)
            
            is_cw = dist_cw <= R[u_idx]
            is_ccw = dist_ccw <= L[u_idx]
            
            if is_cw:
                if is_ccw and len(path_ccw) < len(path_cw):
                    all_paths[(u, v)] = path_ccw
                else:
                    all_paths[(u, v)] = path_cw
            elif is_ccw:
                all_paths[(u, v)] = path_ccw
            else:
                all_paths[(u, v)] = (u, 0, v) # Through center

    return all_paths, L

# --- 3. Linear Programming / Metrizability Check ---

ObjectiveKind = Literal["sum", "max"]
EPSILON = 1e-4  # Gap for strict inequalities

def solve_metric_lp(
    R: List[int],
    objective: ObjectiveKind = "sum"
):
    """
    Builds and solves the LP to check strong metrizability.
    """
    # Generate the system
    path_system, L_vec = generate_wheel_paths(R)
    
    n_rim = len(R)
    vertices = [0] + list(range(1, n_rim + 1))
    
    # Initialize Gurobi
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

    # Variables: d(u, v) for all pairs
    d_vars = {}
    d_exprs = {} # Can hold Vars or LinExprs

    for u, v in combinations(vertices, 2):
        key = ts(u, v)
        # Lower bound 1 ensures positive weights
        var = model.addVar(name=f"d_{key[0]}_{key[1]}", lb=1.0, vtype=GRB.CONTINUOUS)
        d_vars[key] = var
        d_exprs[key] = var

    model.update()

    # Constraints: Path System Consistency
    # For every pair (u, v), and every intermediate node k:
    # If k is on the shortest path: d(u,v) = d(u,k) + d(k,v)
    # If k is NOT on the shortest path: d(u,v) < d(u,k) + d(k,v)  (Strict triangle ineq)
    
    for (u, v), path in path_system.items():
        pair_key = ts(u, v)
        
        # Identify nodes on the path to skip strict check (they get equality check)
        nodes_on_path = set(path)
        
        # 1. Equality constraints for nodes ON the path
        # If path is (u, ..., k, ..., v), then d(u,v) == d(u,k) + d(k,v)
        # Ideally, we just sum the edges along the path.
        path_len_expr = gp.LinExpr()
        for k in range(len(path) - 1):
            edge = ts(path[k], path[k+1])
            path_len_expr += d_vars[edge]
        
        model.addConstr(d_vars[pair_key] == path_len_expr, name=f"PathEq_{u}_{v}")

        # 2. Strict Triangle Inequality for nodes NOT on the path
        # For strong metrizability, any node 'k' not on the geodesic must satisfy strict inequality
        for k in vertices:
            if k not in nodes_on_path:
                # d(u, v) < d(u, k) + d(k, v)
                # Implemented as: d(u, v) + epsilon <= d(u, k) + d(k, v)
                edge_uk = ts(u, k)
                edge_kv = ts(k, v)
                model.addConstr(
                    d_vars[pair_key] + EPSILON <= d_vars[edge_uk] + d_vars[edge_kv],
                    name=f"StrictTri_{u}_{v}_via_{k}"
                )

    # Objective Function
    # We want to minimize weights of the physical graph edges:
    # Hub edges: (0, 1) ... (0, n)
    # Rim edges: (1, 2) ... (n, 1)
    
    physical_edges = []
    # Hub edges (H)
    for i in range(1, n_rim + 1):
        physical_edges.append(ts(0, i))
    # Rim edges (X)
    for i in range(1, n_rim + 1):
        next_i = (i % n_rim) + 1
        physical_edges.append(ts(i, next_i))

    if objective == "sum":
        obj_expr = gp.quicksum(d_vars[e] for e in physical_edges)
        model.setObjective(obj_expr, GRB.MINIMIZE)
    
    elif objective == "max":
        # Introduce auxiliary variable M
        M = model.addVar(name="MaxWeight", vtype=GRB.CONTINUOUS)
        for e in physical_edges:
            model.addConstr(M >= d_vars[e])
        model.setObjective(M, GRB.MINIMIZE)

    # Solve
    model.optimize()

    # Results extraction
    result = {
        "status": model.Status,
        "L_vec": L_vec,
        "weights": {}
    }

    if model.Status == GRB.OPTIMAL:
        result["obj_val"] = model.ObjVal
        for i in range(1, n_rim + 1):
            # H_i corresponds to edge (0, i) (index i-1 in lists usually, but here i)
            # X_i corresponds to edge (i, i+1)
            h_edge = ts(0, i)
            next_i = (i % n_rim) + 1
            x_edge = ts(i, next_i)
            
            # Using 0-based indexing for output display (H0..Hn-1) to match user preference
            result["weights"][f"H{i-1}"] = d_vars[h_edge].X
            result["weights"][f"X{i-1}"] = d_vars[x_edge].X
            
    return result

# --- 4. Main Execution ---

def print_results(R, res):
    print(f"\n{'='*40}")
    print(f"Results for R = {R}")
    print(f"{'='*40}")
    print(f"Derived L Vector: {res['L_vec']}")
    
    if res['status'] == GRB.OPTIMAL:
        print(f"\nStatus: Strong Metric FOUND")
        print(f"Objective Value: {res['obj_val']:.4f}")
        print("\nEdge Weights:")
        # Note: H are the internal (spoke) edges from the center outward, and X are the rim (perimeter) edges
        # Sort keys to print H0, X0, H1, X1...
        sorted_keys = sorted(res['weights'].keys(), key=lambda k: (int(k[1:]), k[0]))
        
        # Print in pairs
        n = len(R)
        for i in range(n):
            h_val = res['weights'][f"H{i}"]
            x_val = res['weights'][f"X{i}"]
            print(f"  i={i}:  H{i} = {h_val:7.4f}  |  X{i} = {x_val:7.4f}")
            
    elif res['status'] == GRB.INFEASIBLE:
        print("\nStatus: INFEASIBLE (Not Strongly Metrizable)")
    else:
        print(f"\nStatus: Code {res['status']}")
    print("-" * 40)

if __name__ == "__main__":

    R1 = [1, 2, 1, 2, 1, 2]    # change here to test different vectors
    res1 = solve_metric_lp(R1, objective="max")
    print_results(R1, res1)
