# wheel_geometry/utils.py
from typing import List

def get_L_from_R(R: List[int]) -> List[int]:
    """Calculates the CCW reach vector L from CW reach vector R."""
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

def get_contracted_indices(n_current: int, contracted_edge_idx: int) -> List[int]:
    """
    Returns a mapping array where map[old_idx] = new_idx.
    Vertices are re-indexed after contraction of (edge_idx, edge_idx+1).
    The contracted node keeps index 'edge_idx'.
    Indices > edge_idx+1 shift down by 1.
    """
    mapping = []
    next_i = (contracted_edge_idx + 1) % n_current
    
    current_new_idx = 0
    for i in range(n_current):
        if i == next_i:
            # This node merges into edge_idx
            mapping.append(contracted_edge_idx) 
        else:
            # Logic to map to new indices (skipping the removed one)
            # If i <= edge_idx, index stays same (mostly)
            # If i > next_i, index decreases by 1
            if i > next_i:
                mapping.append(i - 1)
            else:
                mapping.append(i)
                
    return mapping