import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, permutations
from typing import List, Dict, Tuple, Set, Any

# --- 1. Helper Functions ---

def ts(u: int, v: int) -> Tuple[int, int]:
    """Returns a sorted tuple for undirected edges/deltas."""
    return tuple(sorted((u, v)))

# --- 2. Path System Generation (0-based indexing) ---

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

def generate_wheel_paths_0based(R: List[int]):
    n = len(R)
    L = derive_l_from_r(R, n)
    all_paths = {}
    
    def get_cw_dist_and_path(u, v):
        if v >= u: return v - u, tuple(range(u, v + 1))
        else: return (n - u) + v, tuple(range(u, n)) + tuple(range(0, v + 1))
    
    def get_ccw_dist_and_path(u, v):
        d, p = get_cw_dist_and_path(v, u)
        return d, p[::-1]

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if j == n:
                all_paths[ts(i, j)] = (i, n)
                continue
            if i == n:
                all_paths[ts(i, j)] = (n, j)
                continue
                
            dist_cw, path_cw = get_cw_dist_and_path(i, j)
            dist_ccw, path_ccw = get_ccw_dist_and_path(i, j)
            
            is_cw = dist_cw <= R[i]
            is_ccw = dist_ccw <= L[i]
            
            if is_cw:
                if is_ccw and len(path_ccw) < len(path_cw):
                    all_paths[ts(i, j)] = path_ccw
                else:
                    all_paths[ts(i, j)] = path_cw
            elif is_ccw:
                all_paths[ts(i, j)] = path_ccw
            else:
                all_paths[ts(i, j)] = (i, n, j)

    return all_paths, L

# --- 3. Poset and Linear Extensions (Combinatorics) ---

def get_all_linear_extensions(items: Set[Any], directed_edges: Set[Tuple[Any, Any]]) -> List[List[Any]]:
    in_degree = {item: 0 for item in items}
    graph = {item: [] for item in items}
    
    for u, v in directed_edges:
        if u in items and v in items:
            graph[u].append(v)
            in_degree[v] += 1
            
    results = []
    
    def backtrack(path, available):
        if len(path) == len(items):
            results.append(list(path))
            return
        
        for node in list(available):
            available.remove(node)
            next_available = set(available)
            
            for child in graph[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_available.add(child)
            
            backtrack(path + [node], next_available)
            
            for child in graph[node]:
                in_degree[child] += 1
            available.add(node)
            
    initial_available = set(n for n in items if in_degree[n] == 0)
    backtrack([], initial_available)
    return results

# --- 4. Analyzer Engine ---

EPSILON = 1e-4

def analyze_r_delta_orders(R: List[int]) -> Tuple[bool, List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], int]:
    n = len(R)
    path_system, L = generate_wheel_paths_0based(R)
    
    relevant_deltas = set()
    poset_edges = set() 
    
    for i in range(n):
        for d in range(1, R[i] + 1):
            full_pair = ts(i, (i + d) % n)
            relevant_deltas.add(full_pair)
            
            for dx in range(d + 1):
                for dy in range(dx + 1, d + 1):
                    if dy - dx < d: 
                        sub_pair = ts((i + dx) % n, (i + dy) % n)
                        relevant_deltas.add(sub_pair)
                        if sub_pair != full_pair:
                            poset_edges.add((sub_pair, full_pair))
                            
    all_orders = get_all_linear_extensions(relevant_deltas, poset_edges)
    total_orders = len(all_orders)
    
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

    d_vars = {}
    for u, v in combinations(range(n + 1), 2):
        d_vars[ts(u, v)] = model.addVar(lb=1.0, name=f"d_{u}_{v}")

    for pair, path in path_system.items():
        nodes_on_path = set(path)
        path_expr = gp.LinExpr()
        for k in range(len(path) - 1):
            path_expr += d_vars[ts(path[k], path[k+1])]
        model.addConstr(d_vars[pair] == path_expr)

        for k in range(n + 1):
            if k not in nodes_on_path:
                model.addConstr(d_vars[pair] + EPSILON <= d_vars[ts(pair[0], k)] + d_vars[ts(k, pair[1])])

    delta_vars = {}
    for u, v in relevant_deltas:
        h_u = d_vars[ts(u, n)]
        h_v = d_vars[ts(v, n)]
        dist_uv = d_vars[ts(u, v)]
        delta_vars[ts(u, v)] = model.addVar(lb=-GRB.INFINITY, name=f"delta_{u}_{v}")
        model.addConstr(delta_vars[ts(u, v)] == h_u + h_v - dist_uv)

    model.setObjective(gp.quicksum(d_vars.values()), GRB.MINIMIZE)
    model.update()
    
    invalid_orders = []
    valid_orders = []
    
    for order in all_orders:
        constrs = []
        for idx in range(len(order) - 1):
            pair_large = order[idx]
            pair_small = order[idx + 1]
            c = model.addConstr(delta_vars[pair_large] >= delta_vars[pair_small] + EPSILON)
            constrs.append(c)
            
        model.optimize()
        
        if model.Status == GRB.INFEASIBLE:
            invalid_orders.append(order)
        elif model.Status == GRB.OPTIMAL:
            valid_orders.append(order)
            
        for c in constrs:
            model.remove(c)
        model.update()
        
    return (len(invalid_orders) == 0), invalid_orders, valid_orders, total_orders

# --- 5. Order Miner Engine (In-Memory) ---

def format_delta(pair: Tuple[int, int]) -> str:
    return f"Δ_{pair[0]},{pair[1]}"

def mine_rules_for_orders(good_orders: List[List[Tuple[int, int]]], bad_orders: List[List[Tuple[int, int]]], max_rule_size: int = 4) -> str:
    """Mines the minimal rule that separates good orders from bad orders."""
    if not good_orders or not bad_orders:
        return "Insufficient data to mine rules."
        
    all_deltas = set(good_orders[0])
    num_good = len(good_orders)
    
    all_pairs = list(permutations(all_deltas, 2))
    
    # Find "safe" features (X > Y that NEVER appear in Bad orders)
    safe_features = []
    for x, y in all_pairs:
        is_safe = True
        for order in bad_orders:
            if order.index(x) < order.index(y):
                is_safe = False
                break
        if is_safe:
            safe_features.append((x, y))
            
    feature_coverage = {}
    for feature in safe_features:
        x, y = feature
        covered_indices = set()
        for i, order in enumerate(good_orders):
            if order.index(x) < order.index(y):
                covered_indices.add(i)
        feature_coverage[feature] = covered_indices

    report = []
    for k in range(1, max_rule_size + 1):
        for combo in combinations(safe_features, k):
            combined_coverage = set()
            for feature in combo:
                combined_coverage.update(feature_coverage[feature])
                
            if len(combined_coverage) == num_good:
                good_cond = " OR ".join([f"({format_delta(x)} > {format_delta(y)})" for x, y in combo])
                bad_cond = " AND ".join([f"({format_delta(y)} > {format_delta(x)})" for x, y in combo])
                report.append(f"🎯 FOUND PERFECT RULE (Size {k}):")
                report.append(f"✅ GOOD (Solvable) iff: {good_cond}")
                report.append(f"❌ BAD (Infeasible) iff: {bad_cond}")
                return "\n".join(report)
                
    return f"Could not find a perfect generic rule up to size {max_rule_size}. The rule might involve triplets."

# --- 6. Batch Generator & Runner ---

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
        if canonical in unique_canonicals: continue
        l = derive_l_from_r(r, n)
        if _is_legal_neighborly_vectors(r, l):
            unique_canonicals.add(canonical)
            results.append(r)
            
    return results

def run_comprehensive_delta_analysis(n: int, output_file: str):
    R_list = generate_all_valid_R(n)
    print(f"Starting Comprehensive Hunt. Found {len(R_list)} unique vectors for n={n}.\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"=== COMPREHENSIVE DELTA ORDER ANALYSIS FOR N={n} ===\n\n")
    
    problematic_systems_found = 0
    
    for R in R_list:
        print(f"Testing R = {R} ... ", end="", flush=True)
        is_fine, bad_orders, good_orders, total_orders = analyze_r_delta_orders(R)
        
        if is_fine:
            print(f"All {total_orders} valid delta orders are FEASIBLE. (Safe)")
        else:
            problematic_systems_found += 1
            print(f"FOUND {len(bad_orders)} INFEASIBLE ORDERS! Mining rules...")
            
            # Mine the rule directly from the generated lists
            mined_rules_text = mine_rules_for_orders(good_orders, bad_orders)
            
            # Append everything to the log file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"SYSTEM: R = {R}\n")
                f.write(f"Total consistent orders tested: {total_orders}\n")
                f.write(f"Feasible (Good) orders found: {len(good_orders)}\n")
                f.write(f"Infeasible (Bad) orders found: {len(bad_orders)}\n")
                f.write(f"{'-'*60}\n")
                f.write("MINED RULES FOR THIS SYSTEM:\n")
                f.write(f"{mined_rules_text}\n")
                f.write(f"{'-'*60}\n")
                
                f.write("--- FEASIBLE (GOOD) ORDERS ---\n")
                for idx, order in enumerate(good_orders):
                    order_str = " > ".join([format_delta(pair) for pair in order])
                    f.write(f"Good #{idx+1}: {order_str}\n")
                
                f.write("\n--- INFEASIBLE (BAD) ORDERS ---\n")
                for idx, order in enumerate(bad_orders):
                    order_str = " > ".join([format_delta(pair) for pair in order])
                    f.write(f"Bad #{idx+1}: {order_str}\n")
                    
                f.write(f"{'='*60}\n\n")
                
    if problematic_systems_found == 0:
        print("\nWOW! All vectors passed. Every consistent order is realizable!")
    else:
        print(f"\nFinished. Found {problematic_systems_found} systems with specific order restrictions.")
        print(f"Full details and mined rules saved to: {output_file}")

if __name__ == "__main__":
    TEST_N = 5  
    OUT_FILE = f"comprehensive_delta_analysis_n{TEST_N}.txt"
    run_comprehensive_delta_analysis(TEST_N, OUT_FILE)