import math
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from typing import List, Dict, Tuple, Literal, Set
from tqdm import tqdm

# ==========================================
# 1. Helper Functions
# ==========================================

def ts(u: int, v: int) -> Tuple[int, int]:
    """Returns a sorted tuple for undirected edges, ensuring consistent dictionary keys."""
    return tuple(sorted((u, v)))

def get_L_from_R(R: List[int]) -> List[int]:
    """
    Derives the CCW reach vector (L) from the CW reach vector (R).
    L[i] represents how far back vertex i+1 can reach.
    """
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

def _is_legal_neighborly_vectors(r: List[int], l: List[int]) -> bool:
    """Validates if the generated R and L vectors form a legal path system."""
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

def get_canonical_bracelet(r: List[int], l: List[int]) -> Tuple[int, ...]:
    """
    Returns the lexicographically maximal representation of the system 
    among all rotations (of R) and reflections (rotations of reversed L).
    Used to filter out symmetric duplicate systems.
    """
    n = len(r)
    candidates = []
    
    # Regular rotations
    curr_r = list(r)
    for _ in range(n):
        candidates.append(tuple(curr_r))
        curr_r = curr_r[1:] + curr_r[:1]
        
    # Reflections (rotations of reversed L)
    curr_l_rev = list(l)[::-1]
    for _ in range(n):
        candidates.append(tuple(curr_l_rev))
        curr_l_rev = curr_l_rev[1:] + curr_l_rev[:1]
        
    return max(candidates)

# ==========================================
# 2. Optimized Generator (Pruned Backtracking)
# ==========================================

def generate_all_valid_R(n: int) -> List[List[int]]:
    """
    Generates all unique valid R vectors (canonical bracelets) for a given n.
    Uses an optimized backtracking algorithm with early pruning to vastly reduce search space.
    """
    unique_canonicals: Set[Tuple[int, ...]] = set()
    results = []
    
    def backtrack(k: int, current_r: List[int], current_L: List[int]):
        # Base Case: Vector is full
        if k == n:
            # Final rigorous check required because current_L updates at the boundary
            if _is_legal_neighborly_vectors(current_r, current_L):
                canonical = get_canonical_bracelet(current_r, current_L)
                if canonical not in unique_canonicals:
                    unique_canonicals.add(canonical)
                    results.append(list(current_r))
            return
            
        # Minimum value constraint: values can drop by at most 1
        start = 1 if k == 0 else max(1, current_r[-1] - 1)
        
        # Upper bound pruning constraints
        if k == 0:
            end_val = n - 2
        else:
            # Constraint 1: Wrap-around safety
            wrap_around_limit = current_r[0] + n - k
            # Constraint 2: L vector capacity limit
            l_limit = n - 1 - current_L[k]
            
            end_val = min(n - 2, wrap_around_limit, l_limit)
            
        for val in range(start, end_val + 1):
            # Temporarily apply reach to L vector
            new_L = list(current_L)
            for step in range(1, val + 1):
                target = (k + step) % n
                if step > new_L[target]:
                    new_L[target] = step
                    
            backtrack(k + 1, current_r + [val], new_L)

    initial_L = [0] * n
    
    # Progress bar on the outermost loop (r[0] selection)
    for r0 in tqdm(range(1, n - 1), desc=f"Generating vectors (n={n})", leave=False):
        new_L = list(initial_L)
        for step in range(1, r0 + 1):
            new_L[step % n] = step
            
        backtrack(1, [r0], new_L)
        
    return results

# ==========================================
# 3. Analytic Reduction (O(n) Math Logic)
# ==========================================

def get_regular_reduction_analytic(R: List[int]) -> Tuple[int, int]:
    """
    Computes the reduced system size (m) and the regular jump (r) analytically.
    Uses the precise physical reduction mapping F(x) = (x + R[x]) mod n.
    """
    n = len(R)
    f = [(i + R[i]) % n for i in range(n)]
    
    # Find all periodic points (surviving vertices)
    periodic_points = set()
    for i in range(n):
        curr = i
        visited = []
        while curr not in visited:
            visited.append(curr)
            curr = f[curr]
        
        cycle_start = visited.index(curr)
        periodic_points.update(visited[cycle_start:])
        
    m = len(periodic_points)
    if m == 0:
        return 0, 0
        
    # Calculate Poincaré rotation number (rho)
    start_node = next(iter(periodic_points))
    curr = start_node
    steps = 0
    displacement = 0
    
    while True:
        jump = R[curr]
        displacement += jump
        curr = f[curr]
        steps += 1
        if curr == start_node:
            break
            
    rho = displacement / (n * steps)
    r = round(m * rho)
    
    return m, r

def is_singular_reduction(R: List[int]) -> bool:
    """
    Checks if the system reduces to a regular system with a singular matrix.
    The matrix is singular iff the size of the independent circular sub-systems is even.
    """
    m, r = get_regular_reduction_analytic(R)
    
    if r < 1 or m == 0: 
        return False
        
    return (m // math.gcd(m, r)) % 2 == 0

# ==========================================
# 4. Metrizability Solver (Gurobi LP)
# ==========================================

def generate_wheel_paths(R: List[int]) -> Tuple[Dict[Tuple[int, int], Tuple[int, ...]], List[int]]:
    """Generates the shortest path system for a wheel graph given a CW reach vector R."""
    n = len(R)
    L = get_L_from_R(R)
    
    all_paths = {}
    vertices = [0] + list(range(1, n + 1))

    def get_cw_dist_and_path(u, v):
        if v >= u: return v - u, tuple(range(u, v + 1))
        else: return (n - u) + v, tuple(range(u, n + 1)) + tuple(range(1, v + 1))

    def get_ccw_dist_and_path(u, v):
        d, p = get_cw_dist_and_path(v, u)
        return d, p[::-1]

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            u, v = vertices[i], vertices[j]
            
            if u == 0:
                all_paths[(u, v)] = (0, v)
                continue
            
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
                all_paths[(u, v)] = (u, 0, v) 

    return all_paths, L

ObjectiveKind = Literal["sum", "max"]
EPSILON = 1e-4

def solve_metric_lp(R: List[int], objective: ObjectiveKind = "sum") -> Dict:
    """Builds and solves the LP to check strong metrizability for a specific R."""
    path_system, L_vec = generate_wheel_paths(R)
    n_rim = len(R)
    vertices = [0] + list(range(1, n_rim + 1))
    
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

    d_vars = {}
    for u, v in combinations(vertices, 2):
        key = ts(u, v)
        var = model.addVar(name=f"d_{key[0]}_{key[1]}", lb=1.0, vtype=GRB.CONTINUOUS)
        d_vars[key] = var

    model.update()

    for (u, v), path in path_system.items():
        pair_key = ts(u, v)
        nodes_on_path = set(path)
        
        # Path equations
        path_len_expr = gp.LinExpr()
        for k in range(len(path) - 1):
            edge = ts(path[k], path[k+1])
            path_len_expr += d_vars[edge]
        model.addConstr(d_vars[pair_key] == path_len_expr, name=f"PathEq_{u}_{v}")

        # Strict Triangle Inequalities
        for k in vertices:
            if k not in nodes_on_path:
                edge_uk = ts(u, k)
                edge_kv = ts(k, v)
                model.addConstr(
                    d_vars[pair_key] + EPSILON <= d_vars[edge_uk] + d_vars[edge_kv],
                    name=f"StrictTri_{u}_{v}_via_{k}"
                )

    physical_edges = []
    for i in range(1, n_rim + 1):
        physical_edges.append(ts(0, i)) 
        next_i = (i % n_rim) + 1
        physical_edges.append(ts(i, next_i))

    if objective == "sum":
        obj_expr = gp.quicksum(d_vars[e] for e in physical_edges)
        model.setObjective(obj_expr, GRB.MINIMIZE)
    elif objective == "max":
        M = model.addVar(name="MaxWeight", vtype=GRB.CONTINUOUS)
        for e in physical_edges:
            model.addConstr(M >= d_vars[e])
        model.setObjective(M, GRB.MINIMIZE)

    model.optimize()

    result = {
        "status": model.Status,
        "is_metrizable": model.Status == GRB.OPTIMAL,
        "L_vec": L_vec,
        "obj_val": model.ObjVal if model.Status == GRB.OPTIMAL else None
    }
    return result

# ==========================================
# 5. Orchestrator (The Main Engine)
# ==========================================

def find_counterexamples_in_singular_systems(n: int):
    """
    Executes the entire pipeline for a given n: 
    Generates -> Filters in O(n) -> Solves via LP.
    """
    print(f"\n{'='*60}")
    print(f"--- Starting Search for Counterexamples (n={n}) ---")
    print(f"{'='*60}")
    
    # 1. Generation
    all_R = generate_all_valid_R(n)
    print(f"[*] Generated {len(all_R)} valid canonical vectors.\n")
    
    # 2. Filtering
    singular_R_list = [
        R for R in tqdm(all_R, desc="Filtering singular systems", leave=False) 
        if is_singular_reduction(R)
    ]
    
    percent = (len(singular_R_list) / len(all_R)) * 100 if all_R else 0.0
    print(f"[*] Found {len(singular_R_list)} singular systems ({percent:.2f}% of total).\n")
    
    if not singular_R_list:
        print("    No singular systems found for this n. Moving on.")
        return
        
    # 3. Solving
    counterexamples = []
    for R in tqdm(singular_R_list, desc="Solving LP for singular systems"):
        res = solve_metric_lp(R, objective="sum") 
        
        if not res['is_metrizable']:
            # Using tqdm.write prevents the print from breaking the progress bar visually
            tqdm.write(f"\n    [! ALARM !] Counterexample found: R = {R}")
            counterexamples.append(R)
            
    print(f"\n{'='*60}")
    print("--- Search Complete ---")
    if counterexamples:
        print(f"BINGO! Found {len(counterexamples)} non-metrizable (counterexample) systems:")
        for ce in counterexamples:
            print(f"  -> {ce}")
    else:
        print("All singular systems tested are METRIZABLE.")
        print("No counterexamples found for this n.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test on your desired n sequence
    for test_n in range(14, 16):
        find_counterexamples_in_singular_systems(test_n)