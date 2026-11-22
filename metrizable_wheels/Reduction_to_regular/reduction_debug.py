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
            # הנוסחה שלך למסלול הכי קצר (שומרים עליה!)
            end_ind = (i + current_R[i] + 1) % n
            small_delta_paths.append((start_ind, end_ind))
            
        big_deltas_list.append(big_delta_paths)
        small_deltas_list.append(small_delta_paths)
        
        current_R = reduce_wheel(current_R, i)
        W_list.append(current_R)
        
    return W_list, indexes_reduced, big_deltas_list, small_deltas_list

def pi(x: int, i: int, n: int) -> int:
    return x % (n - 1) if x <= i else (x - 1) % (n - 1)

def generate_raw_equations_strings(W_list, indexes_reduced, big_deltas_list, small_deltas_list) -> List[str]:
    """Generates the equations projected to the REDUCED wheel (k+1 level deltas)"""
    m = len(indexes_reduced)
    equations = []
    for k in range(m):
        if bool(big_deltas_list[k]) or bool(small_deltas_list[k]):
            equations.append(f"--- Step k={k} (Reduced edge {indexes_reduced[k]}) ---")
            i_edge = indexes_reduced[k]
            current_n = len(W_list[k])
            
            for (x, y) in big_deltas_list[k]:
                pi_x = pi(x, i_edge, current_n)
                pi_y = pi(y, i_edge, current_n)
                equations.append(f"t_{k} < Δ^({k+1})_{pi_x},{pi_y}")
                
            for (x, y) in small_deltas_list[k]:
                pi_x = pi(x, i_edge, current_n)
                pi_y = pi(y, i_edge, current_n)
                equations.append(f"Δ^({k+1})_{pi_x},{pi_y} < t_{k}")
    return equations

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
        
    report.append("\n=== Raw inequalities (projected to the reduced wheel) ===")
    raw_equations = generate_raw_equations_strings(W_list, indexes_reduced, big_deltas, small_deltas)
    if not raw_equations:
        report.append("No constraints generated.")
    else:
        report.extend(raw_equations)

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
# Example Usage Panel
# ==========================================
if __name__ == "__main__":
    R_test = [1, 2, 1, 2, 1, 2]
    full_report = analyze_wheel(R_test)
    print(full_report)