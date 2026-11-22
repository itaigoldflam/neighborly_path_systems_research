from typing import List, Tuple

import networkx as nx

from path_systems_lib.theta_ab2 import extract_parameters, get_consistent_systems

# =====================================================================
# Metric Testing Framework
# =====================================================================


def extract_true_R_L(system, a: int, b: int) -> Tuple[List[int], List[int]]:
    R, L, _, _ = extract_parameters(system, a, b)
    return R, L


def compute_metric_weights(true_R: List[int], true_L: List[int], a: int, b: int) -> Tuple[List[float], float, float]:
    N = a + b

    # ==========================================
    # 1. Zone Classification (The "Divide & Conquer")
    # ==========================================
    zones = []
    for i in range(N):
        if i < a:
            # Branch A (Nodes 0 to a-1)
            if i + 1 + true_R[(i + 1) % N] < a:
                zones.append('Av*')  # CW reach cannot hit 'a', must use center. Near 'v'.
            elif i - true_L[i] > 0:
                zones.append('Au*')  # CCW reach cannot hit '0', must use center. Near 'u'.
            else:
                zones.append('Ia')   # Intersection: can reach both ends via rim.
        else:
            # Branch B (Nodes a to N-1)
            # Note: Moving CW increases index towards N. Moving CCW decreases towards a.
            if i + 1 + true_R[(i + 1) % N] < N:
                zones.append('Au_b')  # CW reach cannot hit 'N' (which is 0). Near 'a'.
            elif i - true_L[i] > a:
                zones.append('Av_b')  # CCW reach cannot hit 'a'. Near 'v'.
            else:
                zones.append('Ib')

    # ==========================================
    # 2. Compute Smooth Zone-Based Rim Weights
    # ==========================================
    # We use a base scale to ensure pure regions easily absorb local derivative differences
    C = 10.0
    x = [0.0] * N

    for i in range(N):
        z = zones[i]
        if z in ['Av*', 'Av_b']:
            # Dictated purely by Clockwise constraints
            dR = true_R[(i + 1) % N] - true_R[i]
            x[i] = C + dR + 0.1
        elif z in ['Au*', 'Au_b']:
            # Dictated purely by Counter-Clockwise constraints
            dL = true_L[i] - true_L[(i + 1) % N]
            x[i] = C + dL + 0.1
        else:
            # Intersection regions: Flat, smooth base weight (No artificial jumps!)
            x[i] = C + 0.1

    # Distance helper functions along the rim
    def dist_CW(u, v):
        if u == v:
            return 0.0
        d = 0.0
        curr = u
        while curr != v:
            d += x[curr]
            curr = (curr + 1) % N
        return d

    def dist_CCW(u, v):
        if u == v:
            return 0.0
        d = 0.0
        curr = u
        while curr != v:
            curr = (curr - 1) % N
            d += x[curr]
        return d

    def D_rim(u, v):
        return min(dist_CW(u, v), dist_CCW(u, v))

    # Precompute shortest rim distances to the center's gateways (0 and a)
    d_to_0 = [D_rim(i, 0) for i in range(N)]
    d_to_a = [D_rim(i, a) for i in range(N)]

    # ==========================================
    # 3. Find Delta = Hv - Hu analytically
    # ==========================================
    split_a = None
    for k in range(a):
        # 0 reaches k CW, a reaches k+1 CCW
        if true_R[0] >= k and (a - true_L[a] <= k + 1):
            split_a = k
            if k == true_R[0]:
                break

    split_b = None
    for k in range(a, N):
        ccw_reach_0 = N - true_L[0]
        if true_R[a] >= (k - a) and (ccw_reach_0 <= k + 1 or (ccw_reach_0 == N and k + 1 == N)):
            split_b = k
            if k == a + true_R[a]:
                break

    Deltas = []
    if split_a is not None:
        UB = d_to_a[split_a] - d_to_0[split_a]
        LB = d_to_a[split_a + 1] - d_to_0[split_a + 1]
        Deltas.append((UB + LB) / 2.0)

    if split_b is not None:
        LB = d_to_a[split_b] - d_to_0[split_b]
        UB = d_to_a[(split_b + 1) % N] - d_to_0[(split_b + 1) % N]
        Deltas.append((UB + LB) / 2.0)

    Delta = sum(Deltas) / len(Deltas) if Deltas else 0.0

    # ==========================================
    # 4. Find W_center = Hv + Hu
    # ==========================================
    def get_M(u):
        return min(d_to_0[u] + Delta / 2.0, d_to_a[u] - Delta / 2.0)

    LB_W = 0.0
    UB_W = float('inf')

    for u in range(N):
        # Clockwise
        v_keep = (u + true_R[u]) % N
        req_min = dist_CW(u, v_keep)
        LB_W = max(LB_W, req_min - get_M(u) - get_M(v_keep))

        v_break = (u + true_R[u] + 1) % N
        if dist_CW(u, v_break) <= dist_CCW(u, v_break) + 1e-7:
            req_max = dist_CW(u, v_break)
            UB_W = min(UB_W, req_max - get_M(u) - get_M(v_break))

        # Counter-Clockwise
        v_keep_ccw = (u - true_L[u]) % N
        req_min_ccw = dist_CCW(u, v_keep_ccw)
        LB_W = max(LB_W, req_min_ccw - get_M(u) - get_M(v_keep_ccw))

        v_break_ccw = (u - true_L[u] - 1) % N
        if dist_CCW(u, v_break_ccw) <= dist_CW(u, v_break_ccw) + 1e-7:
            req_max_ccw = dist_CCW(u, v_break_ccw)
            UB_W = min(UB_W, req_max_ccw - get_M(u) - get_M(v_break_ccw))

    if LB_W >= UB_W - 1e-7:
        W_center = LB_W + 0.1
    elif UB_W == float('inf'):
        W_center = LB_W + 5.0
    else:
        W_center = (LB_W + UB_W) / 2.0

    Hv = (W_center + Delta) / 2.0
    Hu = (W_center - Delta) / 2.0

    return x, Hv, Hu


def get_metric_induced_R_L(rim_weights: List[float], w_v: float, w_u: float, a: int, b: int) -> Tuple[List[int], List[int]]:
    """
    Calculates the R and L vectors naturally induced by the given edge weights
    using all-pairs shortest paths.
    """
    N = a + b
    center = N

    # Build weighted graph
    G = nx.Graph()
    for i in range(N):
        G.add_edge(i, (i + 1) % N, weight=rim_weights[i])
    G.add_edge(0, center, weight=w_v)
    G.add_edge(a, center, weight=w_u)

    # Compute all shortest path distances
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    induced_R = [0] * N
    induced_L = [0] * N

    epsilon = 1e-9  # Float precision tolerance

    for i in range(N):
        # Clockwise induction (R)
        current_rim_dist = 0.0
        for step in range(1, N):
            edge_idx = (i + step - 1) % N
            current_rim_dist += rim_weights[edge_idx]
            target = (i + step) % N

            # Keep path if rim distance is <= shortest distance in graph
            if current_rim_dist <= dist[i][target] + epsilon:
                induced_R[i] = step
            else:
                break

        # Counter-clockwise induction (L)
        current_rim_dist = 0.0
        for step in range(1, N):
            edge_idx = (i - step) % N
            current_rim_dist += rim_weights[edge_idx]
            target = (i - step) % N

            if current_rim_dist <= dist[i][target] + epsilon:
                induced_L[i] = step
            else:
                break

    return induced_R, induced_L


# =====================================================================
# Main Execution
# =====================================================================

def main():
    a, b = 5, 4
    print(f"Generating systems for Theta({a},{b},2)...")
    systems, _ = get_consistent_systems(a, b)
    print(f"Testing Metric against {len(systems)} actual path systems...\n")

    failed_count = 0
    for idx, sys in enumerate(systems):
        # Extract the true geometric R, L
        true_R, true_L = extract_true_R_L(sys, a, b)

        # Build the metric
        rim_w, w_v, w_u = compute_metric_weights(true_R, true_L, a, b)

        # See what paths the metric actually creates
        ind_R, ind_L = get_metric_induced_R_L(rim_w, w_v, w_u, a, b)

        # Compare
        if ind_R != true_R or ind_L != true_L:
            failed_count += 1
            if failed_count == 1:
                print(f"--- FIRST FAILURE DETECTED (System #{idx + 1}) ---")
                print(f"True R:    {true_R}")
                print(f"Induced R: {ind_R}")
                print(f"True L:    {true_L}")
                print(f"Induced L: {ind_L}\n")
                print(f"Edge Weights: {rim_w}")
                print(f"Spokes: v={w_v}, u={w_u}\n")

    if failed_count == 0:
        print("SUCCESS! Metric perfectly induces all path systems.")
    else:
        print(f"METRIC FAILED: Incorrectly induced {failed_count} out of {len(systems)} systems.")


if __name__ == "__main__":
    main()
