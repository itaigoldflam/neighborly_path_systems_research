import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Any
import ast
from tqdm import tqdm

# ==========================================
# 1. Graph Reduction Logic
# ==========================================

def get_L_from_R(R: List[int]) -> List[int]:
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

def get_reducible_edges_right(R: List[int]) -> List[int]:
    n = len(R)
    endpoints = {(i + R[i]) % n for i in range(n)}
    return [i for i in range(n) if i not in endpoints]

def get_reducible_edges_left(L: List[int]) -> List[int]:
    n = len(L)
    startpoints = {(i - L[i]) % n for i in range(n)}
    return [i for i in range(n) if (i + 1) % n not in startpoints]

def reduce_wheel(R: List[int], i: int) -> List[int]:
    n = len(R)
    R_red = [0] * (n - 1)
    for v in range(n):
        dist_v_to_i = (i - v) % n
        adjusted_reach = R[v] - 1 if dist_v_to_i < R[v] else R[v]
        mapped_v = v % (n - 1) if v <= i else (v - 1) % (n - 1)
        
        if v == i or v == (i + 1) % n:
            R_red[mapped_v] = max(R_red[mapped_v], adjusted_reach)
        else:
            R_red[mapped_v] = adjusted_reach
    return R_red

def build_system_from_path(R_initial: List[int], reduction_path: List[Tuple[int, str]]):
    W_list = [R_initial]
    indexes_reduced = []
    big_deltas_list = []
    small_deltas_list = []
    directions_used = []
    current_R = R_initial.copy()
    
    for (i, direction) in reduction_path:
        n = len(current_R)
        indexes_reduced.append(i)
        directions_used.append(direction)
        
        old_R = W_list[-1]
        old_L = get_L_from_R(old_R)
        
        big_delta_paths = []
        small_delta_paths = []
        
        if direction == 'R':
            for j in range(old_L[i], -1, -1):
                start_ind = (i - j) % n
                prev_ind = (start_ind - 1) % n
                if old_R[start_ind] > j and old_R[start_ind] > old_R[prev_ind] - 1:
                    end_ind = (start_ind + old_R[start_ind]) % n
                    big_delta_paths.append((start_ind, end_ind))
                    
            if old_R[(i + 1) % n] > old_R[i] - 1:
                start_ind = (i + 1) % n
                end_ind = (i + old_R[i] + 1) % n
                small_delta_paths.append((start_ind, end_ind))
                
        else:
            for j in range(old_R[(i + 1) % n], -1, -1):
                end_ind = (i + 1 + j) % n
                next_ind = (end_ind + 1) % n
                if old_L[end_ind] > j and old_L[end_ind] > old_L[next_ind] - 1:
                    start_ind = (end_ind - old_L[end_ind]) % n
                    big_delta_paths.append((start_ind, end_ind))
            
            if old_L[i] > old_L[(i + 1) % n] - 1:
                start_ind = (i - old_L[(i + 1) % n]) % n
                end_ind = i
                small_delta_paths.append((start_ind, end_ind))
                
        big_deltas_list.append(big_delta_paths)
        small_deltas_list.append(small_delta_paths)
        
        current_R = reduce_wheel(current_R, i)
        W_list.append(current_R)
        
    return W_list, indexes_reduced, big_deltas_list, small_deltas_list, directions_used

def pi(x: int, i: int, n: int) -> int:
    return x % (n - 1) if x <= i else (x - 1) % (n - 1)

def generate_equations_strings(W_list, indexes_reduced, big_deltas_list, small_deltas_list) -> List[str]:
    m = len(indexes_reduced)
    equations = []
    
    def get_projected_path_and_ts(curr_x, curr_y, start_step):
        t_acc = []
        for s in range(start_step, m):
            n_s = len(W_list[s])
            i_s = indexes_reduced[s]
            next_x = pi(curr_x, i_s, n_s)
            next_y = pi(curr_y, i_s, n_s)
            if (next_y - next_x) % (n_s - 1) != (curr_y - curr_x) % n_s:
                t_acc.append(f"t_{s}")
            curr_x = next_x
            curr_y = next_y
        return curr_x, curr_y, t_acc

    for k in range(m):
        if bool(big_deltas_list[k]) or bool(small_deltas_list[k]):
            equations.append(f"--- Step k={k} (Reduced edge {indexes_reduced[k]}) ---")
            current_n = len(W_list[k])
            i_edge = indexes_reduced[k]

            for (x, y) in big_deltas_list[k]:
                pi_x = pi(x, i_edge, current_n)
                pi_y = pi(y, i_edge, current_n)
                final_x, final_y, t_acc = get_projected_path_and_ts(pi_x, pi_y, k + 1)
                t_side = f"t_{k}" + "".join([f" + {t_val}" for t_val in t_acc])
                h_str = f"h_{final_x} + h_{final_y}"
                equations.append(f"big_delta_k{k}_{x}_{y}: {t_side} < {h_str}")
                
            for (x, y) in small_deltas_list[k]:
                pi_x = pi(x, i_edge, current_n)
                pi_y = pi(y, i_edge, current_n)
                final_x, final_y, t_acc = get_projected_path_and_ts(pi_x, pi_y, k + 1)
                t_side = f"t_{k}" + "".join([f" + {t_val}" for t_val in t_acc])
                h_str = f"h_{final_x} + h_{final_y}"
                equations.append(f"small_delta_k{k}_{x}_{y}: {h_str} < {t_side}")
                
    return equations

# ==========================================
# 2. Solver & IIS Extractor
# ==========================================

def solve_system_with_iis(W_list, indexes_reduced, big_deltas_list, small_deltas_list, silent=True) -> Dict[str, Any]:
    m = len(indexes_reduced)
    final_R = W_list[-1]
    final_n = len(final_R)
    
    model = gp.Model("Wheel_Metrizability")
    if silent:
        model.setParam('OutputFlag', 0)

    t_vars = [model.addVar(vtype=GRB.CONTINUOUS, name=f"t_{k}", lb=1.0) for k in range(m)]
    h_vars = [model.addVar(vtype=GRB.CONTINUOUS, name=f"h_{v}", lb=1.0) for v in range(final_n)]

    def get_delta_expr(curr_k: int, x: int, y: int):
        if curr_k == m:
            return h_vars[x] + h_vars[y]
        current_R = W_list[curr_k]
        i_s = indexes_reduced[curr_k]
        n_s = len(current_R)
        next_x = pi(x, i_s, n_s)
        next_y = pi(y, i_s, n_s)
        expr = get_delta_expr(curr_k + 1, next_x, next_y)
        if (next_y - next_x) % (n_s - 1) != (y - x) % n_s:
            expr = expr - t_vars[curr_k]
        return expr

    epsilon = 1e-4 
    for k in range(m):
        current_n = len(W_list[k])
        i_edge = indexes_reduced[k]
        for (x, y) in big_deltas_list[k]:
            pi_x = pi(x, i_edge, current_n)
            pi_y = pi(y, i_edge, current_n)
            delta_expr = get_delta_expr(k + 1, pi_x, pi_y)
            model.addConstr(t_vars[k] <= delta_expr - epsilon, name=f"big_delta_k{k}_{x}_{y}")
        for (x, y) in small_deltas_list[k]:
            pi_x = pi(x, i_edge, current_n)
            pi_y = pi(y, i_edge, current_n)
            delta_expr = get_delta_expr(k + 1, pi_x, pi_y)
            model.addConstr(delta_expr <= t_vars[k] - epsilon, name=f"small_delta_k{k}_{x}_{y}")

    model.setObjective(gp.quicksum(t_vars) + gp.quicksum(h_vars), GRB.MINIMIZE)
    model.optimize()
    
    result = {
        "status": model.status,
        "is_metrizable": model.status == GRB.OPTIMAL,
        "iis_constraints": []
    }
    
    if not result["is_metrizable"] and model.status == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            for c in model.getConstrs():
                if c.IISConstr:
                    result["iis_constraints"].append(c.ConstrName)
        except gp.GurobiError as e:
            result["iis_constraints"].append(f"Failed to compute IIS: {e}")
            
    return result

# ==========================================
# 3. Backtracking Engine (Hunts for Failures)
# ==========================================

def analyze_infeasible_wheel(R_initial: List[int], silent_gurobi: bool = True) -> str:
    """
    Explores the entire tree. If it finds a solution, returns None (we don't care about it).
    If it's proven infeasible, returns a detailed report of the FIRST path tried and its IIS.
    """
    paths_explored = 0
    first_failure_data = None
    
    def dfs(current_R, path: List[Tuple[int, str]]) -> Any:
        nonlocal paths_explored, first_failure_data
        L = get_L_from_R(current_R)
        red_right = get_reducible_edges_right(current_R)
        red_left = get_reducible_edges_left(L)
        
        if not red_right and not red_left:
            paths_explored += 1
            W_list, indexes_reduced, big_deltas, small_deltas, dirs = build_system_from_path(R_initial, path)
            sol = solve_system_with_iis(W_list, indexes_reduced, big_deltas, small_deltas, silent=silent_gurobi)
            
            if sol["is_metrizable"]:
                return True # Found a solution, we can stop and ignore this wheel
                
            if first_failure_data is None:
                # Save the very first path we tried (usually "all right") to show the user the contradiction
                first_failure_data = (path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs)
            return False
            
        for i_r in red_right:
            if dfs(reduce_wheel(current_R, i_r), path + [(i_r, 'R')]): return True
            
        for i_l in red_left:
            if dfs(reduce_wheel(current_R, i_l), path + [(i_l, 'L')]): return True
            
        return False

    is_metrizable = dfs(R_initial, [])
    
    # If the wheel is metrizable somewhere in the tree, we don't want to report it!
    if is_metrizable:
        return None 
        
    # If we got here, it's 100% INFEASIBLE. Let's format the report for the first path.
    path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs = first_failure_data
    
    report = []
    report.append(f"=== PROVEN INFEASIBLE: Wheel R = {R_initial} ===")
    report.append(f"Total combinations explored and failed: {paths_explored}")
    report.append(f"\nAnalyzing a sample failed path to find the contradiction:")
    path_str = " -> ".join([f"Edge {idx}({dir})" for idx, dir in path])
    report.append(path_str)
    
    report.append("\n=== Final linear system for this path ===")
    # Notice I added the constraint names to the strings so you can cross-reference them!
    equations = generate_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    if not equations:
        report.append("No constraints generated.")
    else:
        report.extend(equations)
        
    report.append("\n=== FATAL CONTRADICTION (Gurobi IIS) ===")
    report.append("The following specific equations contradict each other and cannot be satisfied simultaneously:")
    if sol["iis_constraints"]:
        for c_name in sol["iis_constraints"]:
            report.append(f"  * {c_name}")
    else:
        report.append("  (Could not isolate IIS, model might be unbounded instead of structurally infeasible)")
        
    report.append("==================================================\n")
    return "\n".join(report)

# ==========================================
# 4. Generator & Runner
# ==========================================

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
    
    # 1. Regular rotations of R (the system itself)
    curr_r = list(r)
    for _ in range(n):
        candidates.append(tuple(curr_r))
        curr_r = curr_r[1:] + curr_r[:1]
        
    # 2. Reflection: Rotations of L reversed (the mirrored system)
    curr_l_rev = list(l)[::-1]
    for _ in range(n):
        candidates.append(tuple(curr_l_rev))
        curr_l_rev = curr_l_rev[1:] + curr_l_rev[:1]
        
    # Return the lexicographically maximal representation among all
    return max(candidates)

def generate_all_valid_R(n: int) -> List[List[int]]:
    unique_canonicals = set()
    results = []
    
    def generate_r_recursive(current_r):
        if len(current_r) == n:
            if current_r[0] < current_r[-1] - 1: return
            yield list(current_r)
            return
        
        start = 1 if not current_r else max(1, current_r[-1] - 1)
        end = n - 1
        for val in range(start, end):
             yield from generate_r_recursive(current_r + [val])

    for r in generate_r_recursive([]):
        # First, compute L
        l = get_L_from_R(r)
        
        # Check if the system is legal at all
        if not _is_legal_neighborly_vectors(r, l):
            continue
            
        # Now that we have both r and l, find the exact canonical form
        canonical = get_canonical_bracelet(r, l)
        
        # Filter duplicates
        if canonical in unique_canonicals:
            continue
            
        unique_canonicals.add(canonical)
        results.append(r)
        
    return results

def run_infeasible_hunter(n: int, output_filepath: str):
    print(f"Hunting for completely INFEASIBLE wheels for n={n}...")
    R_list = generate_all_valid_R(n)
    print(f"Generated {len(R_list)} unique valid wheels to test.")
    
    total_infeasible = 0
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for idx, R in enumerate(tqdm(R_list, desc="Scanning wheels", unit="wheel")):
            # We don't print every single step to the console so it won't be messy,
            # just a progress indicator.
            # if idx % 10 == 0:
                # print(f"Scanned {idx}/{len(R_list)}...")
                
            report = analyze_infeasible_wheel(R, silent_gurobi=True)
            
            # If report is NOT None, it means the wheel is truly infeasible!
            if report:
                print(f"\n>>> CAUGHT INFEASIBLE WHEEL: {R} <<<")
                f.write(report + "\n\n")
                total_infeasible += 1
                
    print(f"\n--- Hunt Complete ---")
    print(f"Found exactly {total_infeasible} completely infeasible wheels out of {len(R_list)}.")
    print(f"Detailed contradiction reports saved to: {output_filepath}")

if __name__ == "__main__":
    WHEEL_SIZE_N = 9
    OUTPUT_FILE = f"{WHEEL_SIZE_N}_infeasible_only_results.txt"
    
    run_infeasible_hunter(WHEEL_SIZE_N, OUTPUT_FILE)