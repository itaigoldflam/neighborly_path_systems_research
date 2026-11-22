import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from typing import List, Tuple, Dict
import random
from tqdm import tqdm

from path_systems_lib.graph_families import circulant

# ==========================================
# 1. Graph Generation & Bulletproof Consistency
# ==========================================
def get_graph() -> nx.Graph:
    """Returns the Circulant(9, [1, 3]) graph."""
    return circulant(9, [1, 3])

def add_path_and_subpaths(path: List[int], current_sys: Dict[Tuple[int, int], List[int]]) -> Dict[Tuple[int, int], List[int]]:
    """
    Takes a path and the current system.
    Extracts ALL subpaths. If any subpath contradicts the existing system, returns None.
    Otherwise, returns a new dictionary with the path and all its subpaths locked in.
    """
    new_sys = current_sys.copy()
    
    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            u, v = path[i], path[j]
            # Ensure key is always (min, max) for undirected graph representation
            key = (min(u, v), max(u, v))
            sub = path[i:j+1]
            oriented_sub = sub if u < v else sub[::-1]

            if key in new_sys:
                # Check for strict consistency
                if new_sys[key] != oriented_sub and new_sys[key][::-1] != oriented_sub:
                    return None # Contradiction found!
            else:
                new_sys[key] = oriented_sub
                
    return new_sys

# ==========================================
# 2. DFS Generator (Longest-Paths First)
# ==========================================
def generate_strict_diverse_systems(G: nx.Graph, max_systems: int = 50) -> List[Dict[Tuple[int, int], List[int]]]:
    """
    Generates fully consistent neighborly path systems using DFS,
    prioritizing longer paths to encourage Delta > 1.
    """
    nodes = list(G.nodes())
    base_system = {}
    
    # Initialize with direct edges (Neighborly condition)
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        base_system[key] = [min(u, v), max(u, v)]
        
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)) if not G.has_edge(nodes[i], nodes[j])]
    
    all_simple_paths = {}
    for u, v in pairs:
        # Get paths up to length 4.
        valid_paths = list(nx.all_simple_paths(G, u, v, cutoff=4))
        
        # Group by length and shuffle within groups to get diverse systems
        by_length = {}
        for p in valid_paths:
            by_length.setdefault(len(p), []).append(p)
            
        shuffled_paths = []
        # Sort descending to try longest paths first (crucial for Delta > 1)
        for l in sorted(by_length.keys(), reverse=True):
            group = by_length[l]
            random.shuffle(group)
            shuffled_paths.extend(group)
            
        all_simple_paths[(u, v)] = shuffled_paths
        
    valid_systems = []

    def backtrack(pair_idx: int, current_system: Dict[Tuple[int, int], List[int]]):
        # Stop condition
        if len(valid_systems) >= max_systems:
            return
            
        # Reached the end of pairs, we have a complete valid system
        if pair_idx == len(pairs):
            valid_systems.append(current_system.copy())
            return
            
        u, v = pairs[pair_idx]
        key = (min(u, v), max(u, v))
        
        # CRITICAL FIX: If this pair was already defined as a subpath of a previous choice, skip it!
        if key in current_system:
            backtrack(pair_idx + 1, current_system)
            return
            
        for path in all_simple_paths[(u, v)]:
            next_sys = add_path_and_subpaths(path, current_system)
            if next_sys is not None:
                backtrack(pair_idx + 1, next_sys)

    print(f"Generating up to {max_systems} STRICTLY consistent diverse systems...")
    backtrack(0, base_system)
    print(f"Successfully generated {len(valid_systems)} systems.")
    return valid_systems

# ==========================================
# 3. Delta Solver & IIS Ratio Analysis
# ==========================================
def get_path_edges(path: List[int]) -> List[Tuple[int, int]]:
    """Converts a path into a list of sorted edge tuples."""
    return [(min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(len(path)-1)]

def check_alpha_with_analysis(G: nx.Graph, system: Dict[Tuple[int, int], List[int]], alpha: float, compute_iis: bool = False):
    """
    Checks if the system is metrizable for a given alpha.
    If compute_iis is True and it's infeasible, returns the IIS edge ratio.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model(env=env)
    
    edge_vars = { (min(u, v), max(u, v)): model.addVar(lb=1.0, name=f"e_{min(u,v)}_{max(u,v)}") for u, v in G.edges() }
    model.update()
    eps = 1e-4
    
    constr_metadata = {}
    
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            key = (min(u, v), max(u, v))
            chosen_path = system[key]
            chosen_edges = get_path_edges(chosen_path)
            chosen_expr = gp.quicksum(edge_vars[e] for e in chosen_edges)
            
            for alt_idx, alt_path in enumerate(nx.all_simple_paths(G, u, v, cutoff=4)):
                alt_edges = get_path_edges(alt_path)
                if set(alt_edges) == set(chosen_edges):
                    continue
                alt_expr = gp.quicksum(edge_vars[e] for e in alt_edges)
                
                c_name = f"C_{u}_{v}_{alt_idx}"
                model.addConstr(chosen_expr <= alpha * alt_expr - eps, name=c_name)
                
                constr_metadata[c_name] = (len(chosen_edges), len(alt_edges))
                
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return True, None, None
        
    elif model.status == GRB.INFEASIBLE and compute_iis:
        model.computeIIS()
        total_left_edges = 0
        total_right_edges = 0
        
        for c in model.getConstrs():
            if c.IISConstr:
                l_edges, r_edges = constr_metadata[c.ConstrName]
                total_left_edges += l_edges
                total_right_edges += r_edges
                
        ratio = total_left_edges / total_right_edges if total_right_edges > 0 else 0
        return False, ratio, (total_left_edges, total_right_edges)
        
    return False, None, None

# ==========================================
# 4. Main Execution
# ==========================================
def explore_variations():
    G = get_graph()
    
    # Let's generate 50 fully locked and strictly consistent systems.
    # Because of the strict consistency, it explores the tree very carefully.
    systems = generate_strict_diverse_systems(G, max_systems=2000)
    
    high_delta_systems = []
    
    for idx, sys in enumerate(tqdm(systems, desc="Checking Delta for systems")):
        if check_alpha_with_analysis(G, sys, 1.0)[0]:
            continue
            
        low, high = 1.0, 5.0
        while high - low > 1e-4:
            mid = (low + high) / 2.0
            if check_alpha_with_analysis(G, sys, mid)[0]: high = mid
            else: low = mid
            
        delta = high
        if delta > 1.01:
            _, ratio, (l_edges, r_edges) = check_alpha_with_analysis(G, sys, low, compute_iis=True)
            high_delta_systems.append({
                'idx': idx + 1,
                'delta': delta,
                'ratio': ratio,
                'l_edges': l_edges,
                'r_edges': r_edges,
                'system': sys
            })
            
    print("\n" + "="*60)
    print(f" FOUND {len(high_delta_systems)} STRICTLY CONSISTENT SYSTEMS WITH DELTA > 1")
    print("="*60)
    
    high_delta_systems.sort(key=lambda x: x['delta'])
    
    for item in high_delta_systems:
        print(f"\n[System #{item['idx']}]")
        print(f"  Delta(P)  = {item['delta']:.4f}")
        print(f"  IIS Ratio = {item['ratio']:.4f}  (Left Edges: {item['l_edges']} / Right Edges: {item['r_edges']})")
        
        long_paths = []
        for (u, v), path in item['system'].items():
            if not G.has_edge(u, v) and len(path) > 3: # path length > 3 means 3 edges
                long_paths.append(f"    {u}->{v}: {path}")
                
        print("  Preferred Long Paths (Length >= 3):")
        for lp in long_paths:
            print(lp)

if __name__ == "__main__":
    explore_variations()