import multiprocessing
from typing import List, Tuple, Set
from tqdm import tqdm

# ==========================================
# 1. Helper Functions (Vector Legality & Symmetry)
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
# 2. Optimized & Paralleled Generator
# ==========================================

def worker_generate(n: int, r0: int) -> List[List[int]]:
    """
    Worker function to be executed in parallel by different CPU cores.
    Generates all valid combinations starting with a specific r0.
    """
    results = []
    
    def backtrack(k: int, current_r: List[int], current_L: List[int]):
        if k == n:
            if _is_legal_neighborly_vectors(current_r, current_L):
                results.append(list(current_r))
            return
            
        start = max(1, current_r[-1] - 1)
        
        if k == n - 1:
            wrap_around_limit = current_r[0] + 1
            l_limit = n - 1 - current_L[k]
            end_val = min(n - 2, wrap_around_limit, l_limit)
        else:
            wrap_around_limit = current_r[0] + n - k
            l_limit = n - 1 - current_L[k]
            end_val = min(n - 2, wrap_around_limit, l_limit)
            
        for val in range(start, end_val + 1):
            new_L = list(current_L)
            for step in range(1, val + 1):
                target = (k + step) % n
                if step > new_L[target]:
                    new_L[target] = step
            backtrack(k + 1, current_r + [val], new_L)

    initial_L = [0] * n
    new_L = list(initial_L)
    for step in range(1, r0 + 1):
        new_L[step % n] = step
        
    backtrack(1, [r0], new_L)
    return results

def generate_all_valid_R(n: int) -> List[List[int]]:
    """
    Uses Multiprocessing to distribute the search tree across all CPU cores.
    """
    unique_canonicals: Set[Tuple[int, ...]] = set()
    final_results = []
    
    cores = multiprocessing.cpu_count()
    args = [(n, r0) for r0 in range(1, n - 1)]
    
    with multiprocessing.Pool(processes=cores) as pool:
        worker_outputs = list(tqdm(
            pool.starmap(worker_generate, args), 
            total=len(args), 
            desc=f"Parallel Gen (n={n}, {cores} cores)"
        ))
        
    for worker_result in worker_outputs:
        for r in worker_result:
            l = get_L_from_R(r)
            canonical = get_canonical_bracelet(r, l)
            if canonical not in unique_canonicals:
                unique_canonicals.add(canonical)
                final_results.append(r)
                
    return final_results

# ==========================================
# 3. Continuous Metric Verifier
# ==========================================

def check_continuous_metric(R: List[int]) -> Tuple[List[int], List[int], bool]:
    """
    מייצר את המטריקה לפי הנוסחה האנליטית הרציפה שפיתחנו
    ובודק האם היא משחזרת במדויק את וקטור ה-R המקורי.
    """
    n = len(R)
    epsilon = 0.25
    
    # הגדרת המשקלים לפי הנוסחה האנליטית
    h = [0.5 * R[i] + epsilon for i in range(n)]
    e = [1 + 0.5 * (R[(i + 1) % n] - R[i]) for i in range(n)]

    R_prime = []
    for i in range(n):
        current_periph_dist = 0.0
        max_k = 1
        
        # סימולציה של ריצה על ההיקף
        for k in range(1, n):
            target = (i + k) % n
            current_periph_dist += e[(i + k - 1) % n]
            
            spoke_dist = h[i] + h[target]
            
            # עיגול ל-5 ספרות למניעת שגיאות נקודה צפה (Floating Point)
            if round(current_periph_dist, 5) < round(spoke_dist, 5):
                max_k = k
            else:
                break
                
        R_prime.append(max_k)
        
    is_match = (R == R_prime)
    return (R, R_prime, is_match)

# ==========================================
# 4. Orchestrator
# ==========================================

def verify_all_systems(n: int):
    print(f"\n{'='*60}")
    print(f"--- Testing Continuous Metric Formula (n={n}) ---")
    print(f"{'='*60}")
    
    # שלב 1: ייצור כל הוקטורים החוקיים במקביל
    all_R = generate_all_valid_R(n)
    print(f"[*] Generated {len(all_R)} valid canonical vectors.\n")
    
    if not all_R:
        print("    No valid systems found for this n. Moving on.")
        return
        
    print("\n[*] Verifying metrics in PARALLEL...")
    failed_Rs = []
    
    cores = multiprocessing.cpu_count()
    
    # שלב 2: בדיקת המטריקה על כל הוקטורים במקביל
    with multiprocessing.Pool(processes=cores) as pool:
        results_iterator = pool.imap(check_continuous_metric, all_R)
        
        for R, R_prime, is_match in tqdm(results_iterator, total=len(all_R), desc="Verifying metrics"):
            if not is_match:
                failed_Rs.append((R, R_prime))

    # שלב 3: סיכום תוצאות
    print(f"\n{'='*60}")
    print("--- Verification Complete ---")
    if not failed_Rs:
        print(f"SUCCESS: All {len(all_R)} R vectors were perfectly reconstructed!")
        print("The continuous metric construction is valid for all generated systems.")
    else:
        print(f"FAILED: Found {len(failed_Rs)} mismatches.")
        print("First 10 mismatches:")
        for orig, calc in failed_Rs[:10]:
            print(f"  Original R: {orig}  ->  Calculated R: {calc}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # נדרש לתמיכה במקביליות במערכות Windows
    multiprocessing.freeze_support()
    
    # אפשר לשנות את הטווח כדי לבדוק גלגלים גדולים יותר.
    # נתחיל מ-n=6 עד n=9 כדי לבדוק את הגלגלים הבעייתיים שהכרנו.
    for test_n in range(11, 15):
        verify_all_systems(test_n)