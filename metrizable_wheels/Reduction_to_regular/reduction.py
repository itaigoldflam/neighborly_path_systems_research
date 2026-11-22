import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Any
import ast

# ==========================================
# 1. Graph Reduction & Metrizability Logic
# ==========================================

def get_reducible_edges(R: List[int]) -> List[int]:
    n = len(R)
    endpoints = {(i + R[i]) % n for i in range(n)}
    return [i for i in range(n) if i not in endpoints]

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

def get_L_from_R(R: List[int]) -> List[int]:
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

def find_reduction_wheels(R: List[int]):
    W_list = [R]
    indexes_reduced = []
    big_deltas_list = []
    small_deltas_list = []
    current_R = R.copy()
    
    while True:
        reducible_indexes = get_reducible_edges(current_R)
        if not reducible_indexes:
            break
            
        i = reducible_indexes[0]
        n = len(current_R)
        indexes_reduced.append(i)
        
        L = get_L_from_R(current_R)
        
        big_delta_paths = []
        for j in range(L[i], -1, -1):
            start_ind = (i - j) % n
            prev_ind = (start_ind - 1) % n
            if current_R[start_ind] > j and current_R[start_ind] > current_R[prev_ind] - 1:
                end_ind = (start_ind + current_R[start_ind]) % n
                big_delta_paths.append((start_ind, end_ind))
                
        small_delta_paths = []
        if current_R[(i + 1) % n] > current_R[i] - 1:
            start_ind = (i + 1) % n
            end_ind = (i + current_R[i] + 1) % n
            small_delta_paths.append((start_ind, end_ind))
            
        big_deltas_list.append(big_delta_paths)
        small_deltas_list.append(small_delta_paths)
        
        current_R = reduce_wheel(current_R, i)
        W_list.append(current_R)
        
    return W_list, indexes_reduced, big_deltas_list, small_deltas_list

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

def analyze_wheel(R_initial: List[int], silent_gurobi: bool = True) -> str:
    report = []
    report.append(f"=== Analysis for Wheel R = {R_initial} ===")
    
    W_list, indexes_reduced, big_deltas, small_deltas = find_reduction_wheels(R_initial)
    report.append(f"Performed {len(indexes_reduced)} reductions.")
    for k in range(len(indexes_reduced)):
        report.append(f"Step {k}: Reduced edge {indexes_reduced[k]}. New R = {W_list[k+1]}")
        
    report.append("\n=== Final linear system (Translated to h_i) ===")
    equations = generate_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    if not equations:
        report.append("No constraints generated.")
    else:
        report.extend(equations)
        
    report.append("\n=== Gurobi Solution ===")
    sol = solve_system(W_list, indexes_reduced, big_deltas, small_deltas, silent=silent_gurobi)
    
    if sol["is_metrizable"]:
        report.append("Result: Metrizable System Found!")
        report.append("Deleted edge lengths (t_k):")
        for k, val in sol["t_vals"].items():
            report.append(f"  t_{k} = {val}")
        report.append("\nFinal Regular Wheel variables (h_v):")
        for v, val in sol["h_vals"].items():
            report.append(f"  h_{v} = {val}")
    else:
        report.append("Result: System is NOT Metrizable (Infeasible/Unbounded).")
        
    report.append("==================================================\n")
    return "\n".join(report)

# ==========================================
# 2. Path Systems Generator (Updated)
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

def get_canonical_bracelet(seq: List[int]) -> Tuple[int, ...]:
    n = len(seq)
    rotations = []
    curr = list(seq)
    for _ in range(n):
        rotations.append(tuple(curr))
        curr = curr[1:] + curr[:1]
    curr_rev = list(seq)[::-1]
    for _ in range(n):
        rotations.append(tuple(curr_rev))
        curr_rev = curr_rev[1:] + curr_rev[:1]
    return max(rotations)

def generate_all_valid_R(n: int) -> List[List[int]]:
    """Generates all unique valid R vectors (up to symmetry) for a given n."""
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
        canonical = get_canonical_bracelet(r)
        if canonical in unique_canonicals:
            continue
            
        l = get_L_from_R(r)
        if _is_legal_neighborly_vectors(r, l):
            # Removed the identical columns check to keep all valid wheels!
            unique_canonicals.add(canonical)
            results.append(r)
            
    return results

# ==========================================
# 3. Input/Output Handlers & Orchestrator
# ==========================================

def read_R_from_file(filepath: str) -> List[List[int]]:
    """Reads a text file containing lists (e.g., [2, 3, 3...]) and parses them."""
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

def run_batch_analysis(mode: str, output_filepath: str, n: int = None, input_filepath: str = None):
    """
    mode: 'generate' or 'file'
    output_filepath: Where to save the massive log of results.
    n: Used if mode == 'generate'
    input_filepath: Used if mode == 'file'
    """
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

    print(f"Starting analysis. Writing results to '{output_filepath}'...")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for idx, R in enumerate(R_list):
            print(f"Analyzing {idx+1}/{len(R_list)}: {R}")
            report = analyze_wheel(R, silent_gurobi=True)
            f.write(report + "\n\n")
            
    print(f"Done! All results have been saved to {output_filepath}")

# ==========================================
# Example Usage Panel
# ==========================================
if __name__ == "__main__":
    # --- Control Panel ---
    MODE = 'generate' # Change to 'file' to read from a text file
    
    # Variables for 'generate' mode
    WHEEL_SIZE_N = 7
    OUTPUT_FILE = f"{WHEEL_SIZE_N}_analysis_results.txt"

    # Variables for 'file' mode
    INPUT_FILE = "my_vectors.txt" 
    
    # --- Run ---
    # run_batch_analysis(
    #     mode=MODE, 
    #     output_filepath=OUTPUT_FILE, 
    #     n=WHEEL_SIZE_N, 
    #     input_filepath=INPUT_FILE
    # )

    R_original = [2, 2, 2, 3, 2, 2, 2, 3]
    R_opposite = R_original[::-1]
    all_variations_R = [R_original[i:] + R_original[:i] for i in range(len(R_original))] + [R_opposite[i:] + R_opposite[:i] for i in range(len(R_opposite))]
    print(all_variations_R)
    for R in all_variations_R:
        print(analyze_wheel(R, silent_gurobi=True))
