import time
from typing import List, Tuple
from tqdm import tqdm

# ==========================================
# 1. Helper Functions (Common)
# ==========================================

def get_L_from_R(R: List[int]) -> List[int]:
    """Derives the CCW reach vector (L) from the CW reach vector (R)."""
    n = len(R)
    L = [0] * n
    for i in range(n):
        reach = R[i]
        for step in range(1, reach + 1):
            target = (i + step) % n
            L[target] = max(L[target], step)
    return L

def _is_legal_neighborly_vectors(r: List[int], l: List[int]) -> bool:
    """Validates if the generated R and L vectors form a legal path system."""
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
    """Returns the lexicographically maximal representation (handles rotation & reflection)."""
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
# 2. NAIVE Generator (Old version)
# ==========================================

def generate_all_valid_R_naive(n: int) -> List[List[int]]:
    """The original, unoptimized generator for baseline comparison."""
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

    # Wrap the outermost iteration with tqdm is tricky here, so we do it on the generated items
    all_raw = list(generate_r_recursive([]))
    
    for r in tqdm(all_raw, desc=f"Naive Processing n={n}", leave=False):
        l = get_L_from_R(r)
        if not _is_legal_neighborly_vectors(r, l):
            continue
            
        canonical = get_canonical_bracelet(r, l)
        if canonical in unique_canonicals:
            continue
            
        unique_canonicals.add(canonical)
        results.append(r)
        
    return results


# ==========================================
# 3. OPTIMIZED Generator (New version)
# ==========================================

def generate_all_valid_R_optimized(n: int) -> List[List[int]]:
    """
    Optimized generator that uses early pruning to cut off dead branches,
    drastically reducing the search space.
    """
    unique_canonicals = set()
    results = []
    
    def backtrack(k: int, current_r: List[int], current_L: List[int]):
        # Base Case: Vector is full
        if k == n:
            # We still run the final rigorous check because current_L might 
            # change due to the final wrap-around reach.
            if _is_legal_neighborly_vectors(current_r, current_L):
                canonical = get_canonical_bracelet(current_r, current_L)
                if canonical not in unique_canonicals:
                    unique_canonicals.add(canonical)
                    results.append(list(current_r))
            return
            
        # Minimum value constraint (r[k] >= r[k-1] - 1)
        start = 1 if k == 0 else max(1, current_r[-1] - 1)
        
        # Upper bound pruning
        if k == 0:
            end_val = n - 2
        else:
            # Pruning Condition 1: Wrap-around limit. r_k can't be so big that it can't 
            # safely drop down to r_0 + 1 by the end of the array.
            wrap_around_limit = current_r[0] + n - k
            
            # Pruning Condition 2: L constraint. r_k + current_L[k] <= n - 1
            l_limit = n - 1 - current_L[k]
            
            end_val = min(n - 2, wrap_around_limit, l_limit)
            
        for val in range(start, end_val + 1):
            # Temporarily update L vector with the reach of 'val'
            new_L = list(current_L)
            for step in range(1, val + 1):
                target = (k + step) % n
                if step > new_L[target]:
                    new_L[target] = step
                    
            backtrack(k + 1, current_r + [val], new_L)

    initial_L = [0] * n
    
    # We apply tqdm on the first element r_0 to show progress
    for r0 in tqdm(range(1, n - 1), desc=f"Optimized Generation n={n}"):
        new_L = list(initial_L)
        for step in range(1, r0 + 1):
            new_L[step % n] = step
            
        backtrack(1, [r0], new_L)
        
    return results


# ==========================================
# 4. Comparison & Verification Tool
# ==========================================

def compare_generators(n: int):
    print(f"\n{'='*50}")
    print(f"--- Running Benchmark for n = {n} ---")
    
    # 1. Run Naive
    t0 = time.time()
    naive_res = generate_all_valid_R_naive(n)
    t1 = time.time()
    naive_time = t1 - t0
    print(f"Naive Generation:     {len(naive_res)} valid systems in {naive_time:.4f} seconds")
    
    # 2. Run Optimized
    t2 = time.time()
    opt_res = generate_all_valid_R_optimized(n)
    t3 = time.time()
    opt_time = t3 - t2
    print(f"Optimized Generation: {len(opt_res)} valid systems in {opt_time:.4f} seconds")
    
    # 3. Verify Equality
    # We map lists to tuples so we can easily compare sets
    set_naive = set(map(tuple, naive_res))
    set_opt = set(map(tuple, opt_res))
    
    if set_naive == set_opt:
        print("\nSUCCESS! Both generators produced the exact same valid systems.")
        speedup = naive_time / opt_time if opt_time > 0 else 0
        print(f"Performance Gain: Optimized is {speedup:.2f}x faster!")
    else:
        print("\nERROR! The generators produced different results.")
        missing = set_naive - set_opt
        extra = set_opt - set_naive
        if missing: print(f"Missing in optimized: {missing}")
        if extra: print(f"Extra in optimized: {extra}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # נתחיל לבדוק על n-ים קטנים כדי לוודא תקינות
    for test_n in [11]:
        compare_generators(test_n)