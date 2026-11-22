import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Any
import ast
import math


# ==========================================
# 1. Graph Reduction & Metrizability Logic
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
    """Reduces edge (i, i+1) physically from the wheel."""
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
    """
    Applies a specific sequence of (edge_index, direction) reductions 
    and builds the corresponding inequalities.
    """
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
            # === רדוקציה ימנית ===
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
            # === רדוקציה שמאלית ===
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
                equations.append(f"{t_side} < {h_str}")
                
            for (x, y) in small_deltas_list[k]:
                pi_x = pi(x, i_edge, current_n)
                pi_y = pi(y, i_edge, current_n)
                final_x, final_y, t_acc = get_projected_path_and_ts(pi_x, pi_y, k + 1)
                t_side = f"t_{k}" + "".join([f" + {t_val}" for t_val in t_acc])
                h_str = f"h_{final_x} + h_{final_y}"
                equations.append(f"{h_str} < {t_side}")
                
    return equations

def solve_system(W_list, indexes_reduced, big_deltas_list, small_deltas_list, silent=True) -> Dict[str, Any]:
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
            model.addConstr(t_vars[k] <= delta_expr - epsilon)
        for (x, y) in small_deltas_list[k]:
            pi_x = pi(x, i_edge, current_n)
            pi_y = pi(y, i_edge, current_n)
            delta_expr = get_delta_expr(k + 1, pi_x, pi_y)
            model.addConstr(delta_expr <= t_vars[k] - epsilon)

    model.setObjective(gp.quicksum(t_vars) + gp.quicksum(h_vars), GRB.MINIMIZE)
    model.optimize()
    
    result = {
        "status": model.status,
        "is_metrizable": model.status == GRB.OPTIMAL,
        "t_vals": {},
        "h_vals": {}
    }
    
    if result["is_metrizable"]:
        result["t_vals"] = {k: round(t_vars[k].X, 4) for k in range(m)}
        result["h_vals"] = {v: round(h_vars[v].X, 4) for v in range(final_n)}
        
    return result

# ==========================================
# 2. Backtracking Engine (Fully Combinatorial)
# ==========================================

def analyze_wheel_with_backtrack(R_initial: List[int], silent_gurobi: bool = True) -> str:
    """
    Explores the entire tree of possible reductions.
    At each step, branches on ALL reducible edges in ALL valid directions.
    """
    paths_explored = 0
    
    def dfs(current_R, path: List[Tuple[int, str]]) -> Any:
        nonlocal paths_explored
        L = get_L_from_R(current_R)
        red_right = get_reducible_edges_right(current_R)
        red_left = get_reducible_edges_left(L)
        
        # Base case: No reducible edges left -> Fully reduced regular wheel
        if not red_right and not red_left:
            paths_explored += 1
            # Build constraints and solve for this specific combinatorial path
            W_list, indexes_reduced, big_deltas, small_deltas, dirs = build_system_from_path(R_initial, path)
            sol = solve_system(W_list, indexes_reduced, big_deltas, small_deltas, silent=silent_gurobi)
            
            if sol["is_metrizable"]:
                return (path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs)
            return None
            
        # Recursive Step: Iterate over all possible choices to branch the tree
        
        # Branch 1: All possible RIGHT reductions
        for i_r in red_right:
            res = dfs(reduce_wheel(current_R, i_r), path + [(i_r, 'R')])
            if res: return res
            
        # Branch 2: All possible LEFT reductions
        for i_l in red_left:
            res = dfs(reduce_wheel(current_R, i_l), path + [(i_l, 'L')])
            if res: return res
            
        # If all branches from this point fail (or if there are no branches)
        return None

    # Start DFS
    result = dfs(R_initial, [])
    
    # Formatter
    report = []
    report.append(f"=== Combinatorial Backtracking Analysis for Wheel R = {R_initial} ===")
    report.append(f"Total paths explored in tree: {paths_explored}")
    
    if result is None:
        report.append("\nResult: System is PROVEN NOT METRIZABLE.")
        report.append("All possible combinations and orders of Left/Right reductions resulted in Infeasible models.")
        report.append("==================================================\n")
        return "\n".join(report)
        
    path, sol, W_list, indexes_reduced, big_deltas, small_deltas, dirs = result
    
    # Format the path for human reading
    path_str = " -> ".join([f"Edge {idx}({dir})" for idx, dir in path])
    report.append(f"\nSUCCESS! Metrizable sequence found:\n{path_str}")
    
    report.append(f"\nPerformed {len(indexes_reduced)} reductions.")
    for k in range(len(indexes_reduced)):
        report.append(f"Step {k} [{dirs[k]}]: Reduced edge {indexes_reduced[k]}. New R = {W_list[k+1]}")
        
    report.append("\n=== Final linear system (Translated to h_i) ===")
    equations = generate_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    if not equations:
        report.append("No constraints generated.")
    else:
        report.extend(equations)
        
    report.append("\n=== Gurobi Solution ===")
    report.append("Deleted edge lengths (t_k):")
    for k, val in sol["t_vals"].items():
        report.append(f"  t_{k} = {val}")
    report.append("\nFinal Regular Wheel variables (h_v):")
    for v, val in sol["h_vals"].items():
        report.append(f"  h_{v} = {val}")
        
    report.append("==================================================\n")
    return "\n".join(report)

# ==========================================
# 3. Path Systems Generator 
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

# ==========================================
# 4. Input/Output Handlers & Orchestrator
# ==========================================

def read_R_from_file(filepath: str) -> List[List[int]]:
    vectors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                vec = ast.literal_eval(line)
                if isinstance(vec, list):
                    vectors.append(vec)
            except Exception as e:
                print(f"Skipping line due to parse error: {line} -> {e}")
    return vectors

def run_batch_analysis_with_backtrack(mode: str, output_filepath: str, n: int = None, input_filepath: str = None):
    R_list = []
    
    if mode == 'generate':
        if not n:
            print("Error: 'n' must be provided for generate mode.")
            return
        print(f"Generating valid R vectors for n={n}...")
        R_list = generate_all_valid_R(n)
        print(f"Generated {len(R_list)} unique vectors.")
        
    elif mode == 'file':
        if not input_filepath:
            print("Error: 'input_filepath' must be provided for file mode.")
            return
        print(f"Reading vectors from {input_filepath}...")
        R_list = read_R_from_file(input_filepath)
        print(f"Loaded {len(R_list)} vectors.")
    else:
        print("Invalid mode selected.")
        return

    if not R_list:
        print("No vectors to analyze. Exiting.")
        return

    print(f"Starting Backtracking analysis. Writing results to '{output_filepath}'...")
    
    total_metrizable = 0
    total_infeasible = 0
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for idx, R in enumerate(R_list):
            print(f"Analyzing {idx+1}/{len(R_list)}: {R} ...", end="", flush=True)
            report = analyze_wheel_with_backtrack(R, silent_gurobi=True)
            f.write(report + "\n\n")
            
            if "PROVEN NOT METRIZABLE" in report:
                print(" Infeasible!")
                total_infeasible += 1
            else:
                print(" Solved!")
                total_metrizable += 1
                
    print(f"\n--- Batch Run Complete ---")
    print(f"Total Metrizable: {total_metrizable}")
    print(f"Total Infeasible: {total_infeasible}")
    print(f"All full reports saved to: {output_filepath}")




# ==========================================
# 1. Analytic Reduction Functions (Poincaré)
# ==========================================

def get_regular_reduction_analytic(R: List[int]) -> Tuple[int, int]:
    """
    Computes the reduced system size (m) and the regular jump (r) analytically.
    Uses the mapping F(x) = (x + R[x]) mod n.
    """
    n = len(R)
    
    # 1. The accurate mapping F(x)
    f = [(i + R[i]) % n for i in range(n)]
    
    # 2. Find all periodic points (these are the exact surviving vertices)
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
        
    # 3. Calculate Poincaré rotation number (rho) on one of the cycles
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
    
    # 4. Extract r
    r = round(m * rho)
    
    return m, r


def get_physical_reduced_system(R_initial: List[int]) -> List[int]:
    """
    Runs the physical backtracking logic to find the final reduced system.
    Returns the final regular R vector, or None if no sequence works.
    Note: We just need to find *any* valid reduction path to get the final shape.
    """
    def dfs(current_R):
        L = get_L_from_R(current_R)
        red_right = get_reducible_edges_right(current_R)
        red_left = get_reducible_edges_left(L)
        
        # Base case: Regular system reached
        if not red_right and not red_left:
            return current_R
            
        # Try right reductions
        for i_r in red_right:
            res = dfs(reduce_wheel(current_R, i_r))
            if res: return res
            
        # Try left reductions
        for i_l in red_left:
            res = dfs(reduce_wheel(current_R, i_l))
            if res: return res
            
        return None

    return dfs(R_initial)

def verify_analytic_reduction(n: int):
    """
    Generates all valid systems for n, physically reduces them, and compares 
    the resulting 'm' and 'r' to the analytic values derived from Poincaré.
    """
    print(f"Generating valid R vectors for n={n}...")
    R_list = generate_all_valid_R(n)
    print(f"Generated {len(R_list)} unique vectors. Starting verification...\n")
    
    mismatches = 0
    successful = 0
    
    for idx, R in enumerate(R_list):
        # 1. Physical Reduction
        physical_final_R = get_physical_reduced_system(R)
        
        if physical_final_R is None:
            print(f"Warning: R={R} could not be physically reduced.")
            continue
            
        # The final system is regular: length is 'm' and all elements are 'r'
        physical_m = len(physical_final_R)
        physical_r = physical_final_R[0] 
        
        # 2. Analytic Reduction
        analytic_m, analytic_r = get_regular_reduction_analytic(R)
        
        # 3. Compare both m and r
        if physical_r != analytic_r or physical_m != analytic_m:
            print(f"MISMATCH for R={R}: Physical (m={physical_m}, r={physical_r}), Analytic (m={analytic_m}, r={analytic_r})")
            mismatches += 1
        else:
            successful += 1
            
        if (idx + 1) % 100 == 0:
            print(f"Checked {idx + 1}/{len(R_list)}...")
            
    print(f"\n--- Verification Complete for n={n} ---")
    print(f"Total checked: {successful + mismatches}")
    print(f"Successful matches: {successful}")
    print(f"Mismatches: {mismatches}")
    
    if mismatches == 0:
        print("\nSUCCESS: The analytic formula matches the physical reduction in 100% of cases!")

    
def is_singular_reduction(R: List[int]) -> bool:
    """
    Checks if the system reduces to a singular regular system.
    """
    m, r = get_regular_reduction_analytic(R)
    
    if r < 1 or m == 0: 
        return False
        
    return (m // math.gcd(m, r)) % 2 == 0

def find_counterexamples_in_singular_systems(n: int):
    print(f"\n{'='*50}")
    print(f"--- Starting Search for Counterexamples (n={n}) ---")
    print(f"{'='*50}")
    
    # 1. Generate all valid R vectors
    print(f"[1/3] Generating all valid R vectors for n={n}...")
    all_R = generate_all_valid_R(n)
    print(f"      Found {len(all_R)} valid canonical vectors.")
    
    # 2. Filter using our new O(n) analytic function
    print("\n[2/3] Filtering for systems with SINGULAR reductions...")
    singular_R_list = [R for R in all_R if is_singular_reduction(R)]
    
    if len(all_R) > 0:
        percent = (len(singular_R_list) / len(all_R)) * 100
    else:
        percent = 0.0
        
    print(f"      Found {len(singular_R_list)} singular systems ({percent:.2f}% of total).")
    
    if not singular_R_list:
        print("\n      No singular systems found for this n. Exiting.")
        return
        
    # 3. Run LP solver strictly on the filtered list
    print("\n[3/3] Running Gurobi LP solver on singular systems...")
    counterexamples = []
    
    for idx, R in enumerate(singular_R_list):
        # run the LP solver you wrote earlier
        res = solve_metric_lp(R, objective="sum") 
        
        if res['status'] == GRB.INFEASIBLE:
            print(f"\n      [! ALARM !] Counterexample found: R = {R}")
            counterexamples.append(R)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(singular_R_list):
            print(f"      Tested {idx + 1}/{len(singular_R_list)}...")
            
    print(f"\n{'='*50}")
    print("--- Search Complete ---")
    if counterexamples:
        print(f"BINGO! Found {len(counterexamples)} non-metrizable (counterexample) systems:")
        for ce in counterexamples:
            print(f"  -> {ce}")
    else:
        print("All singular systems tested are METRIZABLE.")
        print("No counterexamples found for this n.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # עכשיו כשהסינון הוא מיידי, אפשר להריץ על n-ים גדולים הרבה יותר מהר!
    for test_n in range(8, 13):
        find_counterexamples_in_singular_systems(test_n)
