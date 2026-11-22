from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# This script checks the family of path systems that we reduce to when we delete edges with epsilon-small weights.
# The code enumerates all path systems in the n-wheel, and prints (to a text file or the terminal) all the path systems it finds.
# This can be used to explicitly study the reduced family after "epsilon-deletion" operations.

# --- Validator & Helper Functions (As defined previously) ---

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

def derive_l_from_r(r: List[int], n: int) -> List[int]:
    l = [0] * n
    for i in range(n):
        reach = r[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            if step > l[target]:
                l[target] = step
    return l

def check_identical_columns(M, n):
    for j in range(n):
        next_j = (j + 1) % n
        cols_match = True
        for i in range(n):
            if i == j or i == next_j: continue
            if M[i, j] != M[i, next_j]:
                cols_match = False
                break
        if cols_match: return True
    return False

# --- NEW: Canonical Form Logic (Symmetry Filter) ---

def get_canonical_bracelet(seq: List[int]) -> Tuple[int, ...]:
    """
    Returns the lexicographically largest sequence among all 
    rotations and reversals of the input sequence.
    This effectively treats the sequence as a 'bracelet'.
    """
    n = len(seq)
    rotations = []
    
    # 1. Generate all cyclic shifts
    curr = list(seq)
    for _ in range(n):
        rotations.append(tuple(curr))
        curr = curr[1:] + curr[:1]  # Rotate left
        
    # 2. Generate all cyclic shifts of the reversed sequence
    curr_rev = list(seq)[::-1]
    for _ in range(n):
        rotations.append(tuple(curr_rev))
        curr_rev = curr_rev[1:] + curr_rev[:1]
        
    return max(rotations)

# --- Main Generator ---

def generate_unique_matrices(n):
    unique_canonicals = set()
    results = []
    
    # Pruning generator for r
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
        # 1. Symmetry Filter First (Optimization)
        # Check if we already have this canonical form
        canonical = get_canonical_bracelet(r)
        if canonical in unique_canonicals:
            continue
            
        # 2. Derive & Validate
        l = derive_l_from_r(r, n)
        if _is_legal_neighborly_vectors(r, l):
            
            # 3. Irreducibility Check
            M = np.zeros((n, n), dtype=int)
            for i in range(n):
                for step in range(1, r[i] + 1):
                    M[i, (i + step) % n] = 1
                for step in range(1, l[i] + 1):
                    M[i, (i - step) % n] = -1
            
            if not check_identical_columns(M, n):
                # Found a new unique valid matrix
                unique_canonicals.add(canonical)
                # Note: We store the specific rotation 'r' that was generated, 
                # but we track uniqueness via 'canonical'.
                results.append((r, l, M))
                
    return results

def plot_matrices_grid(results, n_val):
    if not results:
        print(f"No matrices found for N={n_val}")
        return

    count = len(results)
    cols = 5
    rows = (count // cols) + (1 if count % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(f"Unique Irreducible Matrices (mod Symmetry) for N={n_val}", fontsize=16)
    
    axes_flat = axes.flatten() if count > 1 else [axes]
    cmap = ListedColormap(['#FF9999', '#F0F0F0', '#9999FF'])

    for idx, (r, l, M) in enumerate(results):
        ax = axes_flat[idx]
        ax.matshow(M, cmap=cmap, vmin=-1, vmax=1)
        # Displaying the vector as title
        ax.set_title(str(r), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(count, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# --- Execution ---

def main(n_min: int, n_max: int, to_graphs: bool) -> None:
    """
    Run the generator for all n from n_min to n_max (inclusive).
    to_graphs: if True, display matrices as graphs; if False, write to text files.
    """
    for n in range(n_min, n_max + 1):
        print(f"Generating for N={n}...")
        matrices = generate_unique_matrices(n)
        print(f"Found {len(matrices)} unique matrices (up to symmetry).")
        if len(matrices) > 0:
            if to_graphs:
                plot_matrices_grid(matrices, n)
            else:
                filename = f"{n}-unique_matrices.txt"
                with open(filename, 'w') as f:
                    for i in matrices:
                        f.write(str(i[0]) + '\n')
                print(f"Written to {filename}")
        print("-" * 30)


if __name__ == "__main__":
    n_min = 5
    n_max = 5
    to_graphs = True  # True = plot, False = write to files
    main(n_min, n_max, to_graphs)