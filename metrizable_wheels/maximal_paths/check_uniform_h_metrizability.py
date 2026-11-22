import math
import multiprocessing
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from typing import List, Dict, Tuple, Literal, Set
from tqdm import tqdm

# ==========================================
# 1. Helper Functions
# ==========================================

def ts(u: int, v: int) -> Tuple[int, int]:
    return tuple(sorted((u, v)))

def get_L_from_R(R: List[int]) -> List[int]:
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

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

def get_canonical_bracelet(r: List[int], l: List[int]) -> Tuple[int, ...]:
    n = len(r)
    candidates = []
    
    curr_r = list(r)
    for _ in range(n):
        candidates.append(tuple(curr_r))
        curr_r = curr_r[1:] + curr_r[:1]
        
    curr_l_rev = list(l)[::-1]
    for _ in range(n):
        candidates.append(tuple(curr_l_rev))
        curr_l_rev = curr_l_rev[1:] + curr_l_rev[:1]
        
    return max(candidates)

# ==========================================
# 2. Optimized & Paralleled Generator
# ==========================================

def worker_generate(n: int, r0: int) -> List[List[int]]:
    results = []
    
    def backtrack(k: int, current_r: List[int], current_L: List[int]):
        if k == n:
            if _is_legal_neighborly_vectors(current_r, current_L):
                results.append(list(current_r))
            return
            
        start = max(1, current_r[-1] - 1)
        
        if k == n - 1:
            wrap_around_limit = current_r[0] + 1
            l_limit = n - 1 - current_L[k]
            end_val = min(n - 2, wrap_around_limit, l_limit)
        else:
            wrap_around_limit = current_r[0] + n - k
            l_limit = n - 1 - current_L[k]
            end_val = min(n - 2, wrap_around_limit, l_limit)
            
        for val in range(start, end_val + 1):
            new_L = list(current_L)
            for step in range(1, val + 1):
                target = (k + step) % n
                if step > new_L[target]:
                    new_L[target] = step
            backtrack(k + 1, current_r + [val], new_L)

    initial_L = [0] * n
    new_L = list(initial_L)
    for step in range(1, r0 + 1):
        new_L[step % n] = step
        
    backtrack(1, [r0], new_L)
    return results

def generate_all_valid_R(n: int) -> List[List[int]]:
    unique_canonicals: Set[Tuple[int, ...]] = set()
    final_results = []
    
    cores = multiprocessing.cpu_count()
    args = [(n, r0) for r0 in range(1, n - 1)]
    
    with multiprocessing.Pool(processes=cores) as pool:
        worker_outputs = list(tqdm(
            pool.starmap(worker_generate, args), 
            total=len(args), 
            desc=f"Parallel Gen (n={n}, {cores} cores)"
        ))
        
    for worker_result in worker_outputs:
        for r in worker_result:
            l = get_L_from_R(r)
            canonical = get_canonical_bracelet(r, l)
            if canonical not in unique_canonicals:
                unique_canonicals.add(canonical)
                final_results.append(r)
                
    return final_results

# ==========================================
# 3. Analytic Reduction (O(n) Math Logic)
# ==========================================

def get_regular_reduction_analytic(R: List[int]) -> Tuple[int, int]:
    n = len(R)
    f = [(i + R[i]) % n for i in range(n)]
    
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
    if m == 0: return 0, 0
        
    start_node = next(iter(periodic_points))
    curr = start_node
    steps = 0
    displacement = 0
    
    while True:
        jump = R[curr]
        displacement += jump
        curr = f[curr]
        steps += 1
        if curr == start_node: break
            
    rho = displacement / (n * steps)
    r = round(m * rho)
    return m, r

def is_singular_reduction(R: List[int]) -> bool:
    m, r = get_regular_reduction_analytic(R)
    if r < 1 or m == 0: return False
    return (m // math.gcd(m, r)) % 2 == 0

# ==========================================
# 4. Metrizability Solver (Gurobi LP)
# ==========================================

def generate_wheel_paths(R: List[int]) -> Tuple[Dict[Tuple[int, int], Tuple[int, ...]], List[int]]:
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
                if is_ccw and len(path_ccw) < len(path_cw): all_paths[(u, v)] = path_ccw
                else: all_paths[(u, v)] = path_cw
            elif is_ccw: all_paths[(u, v)] = path_ccw
            else: all_paths[(u, v)] = (u, 0, v) 

    return all_paths, L

def solve_metric_lp(R: List[int], objective: Literal["sum", "max"] = "sum") -> Dict:
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

    # ====== הוספת משתנה ואילוץ לחישורים אחידים ======
    h_uniform = model.addVar(name="h_uniform", lb=1.0, vtype=GRB.CONTINUOUS)
    for i in range(1, n_rim + 1):
        model.addConstr(d_vars[ts(0, i)] == h_uniform, name=f"UniformSpoke_{i}")
    # ====================================================

    model.update()

    EPSILON = 1e-4
    for (u, v), path in path_system.items():
        pair_key = ts(u, v)
        nodes_on_path = set(path)
        
        path_len_expr = gp.LinExpr()
        for k in range(len(path) - 1):
            edge = ts(path[k], path[k+1])
            path_len_expr += d_vars[edge]
        model.addConstr(d_vars[pair_key] == path_len_expr, name=f"PathEq_{u}_{v}")

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
        physical_edges.append(ts(i, (i % n_rim) + 1))

    if objective == "sum":
        model.setObjective(gp.quicksum(d_vars[e] for e in physical_edges), GRB.MINIMIZE)
    elif objective == "max":
        M = model.addVar(name="MaxWeight", vtype=GRB.CONTINUOUS)
        for e in physical_edges: model.addConstr(M >= d_vars[e])
        model.setObjective(M, GRB.MINIMIZE)

    model.optimize()

    return {
        "status": model.Status,
        "is_metrizable": model.Status == GRB.OPTIMAL,
        "L_vec": L_vec,
        "obj_val": model.ObjVal if model.Status == GRB.OPTIMAL else None,
        "R": R
    }

# ==========================================
# 5. Orchestrator
# ==========================================

def find_counterexamples_in_systems(n: int):
    print(f"\n{'='*60}")
    print(f"--- Starting Search for Counterexamples with UNIFORM SPOKES (n={n}) ---")
    print(f"{'='*60}")
    
    all_R = generate_all_valid_R(n)
    print(f"[*] Generated {len(all_R)} valid canonical vectors.\n")
    
    # במידה ותרצה לבדוק רק סינגולריות, שים כאן תנאי:
    # systems_to_test = [R for R in all_R if is_singular_reduction(R)]
    # החלטתי כרגע לבדוק על *כל* המערכות כדי לראות אם הטענה הגלובלית שלך מחזיקה:
    systems_to_test = all_R
    
    if not systems_to_test:
        print("    No systems to test. Moving on.")
        return
        
    print("\n[3/3] Running Gurobi LP solver in PARALLEL...")
    counterexamples = []
    
    cores = multiprocessing.cpu_count()
    
    # Run the Gurobi solver in parallel
    with multiprocessing.Pool(processes=cores) as pool:
        results_iterator = pool.imap(solve_metric_lp, systems_to_test)
        
        for res in tqdm(results_iterator, total=len(systems_to_test), desc=f"Solving ({cores} cores)"):
            if not res['is_metrizable']:
                failed_R = res['R']
                tqdm.write(f"\n    [! ALARM !] Non-metrizable with uniform spokes: R = {failed_R}")
                counterexamples.append(failed_R)

    print(f"\n{'='*60}")
    print("--- Search Complete ---")
    if counterexamples:
        print(f"BINGO! Found {len(counterexamples)} systems that CANNOT be metrized with uniform spokes:")
        for ce in counterexamples: print(f"  -> {ce}")
    else:
        print("SUCCESS: All tested systems are METRIZABLE even with UNIFORM SPOKES.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # Run the tests! (שיניתי ל-nים קטנים יותר לצורך בדיקה מהירה, תשנה בחזרה כרצונך)
    for test_n in range(6, 12):
        find_counterexamples_in_systems(test_n)