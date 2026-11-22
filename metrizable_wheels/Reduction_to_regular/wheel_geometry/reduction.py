import numpy as np
from typing import List, Tuple, Dict, Optional
from wheel_geometry.utils import get_L_from_R

class WheelReductionPlanner:
    def __init__(self, R_initial: List[int]):
        self.original_R = list(R_initial)
        self.history = [] 

    def find_reducible_edge(self, R: List[int], debug=False) -> Tuple[Optional[int], str]:
        n = len(R)
        L = get_L_from_R(R)
        
        # Flags for blocking
        is_cw_end = [False] * n   # Edge leading into a CW path end
        is_cw_start = [False] * n # Edge leading out of a CW path start
        
        # Analyze Endpoints
        for k in range(n):
            # CW Path: k -> ... -> end_v
            # The edge (end_v-1, end_v) is the "End Edge".
            end_v = (k + R[k]) % n
            last_edge = (end_v - 1) % n
            is_cw_end[last_edge] = True
            
            # The edge (k, k+1) is the "Start Edge".
            # We mark it if the path actually takes a step (R > 0)
            if R[k] > 0:
                is_cw_start[k] = True

        # --- Scoring System ---
        # Lower score is better.
        # Penalty for being an END (CW Stop): High (we really want to avoid this)
        # Penalty for being a START (CCW Stop): Low (less critical)
        
        candidates = []
        
        for i in range(n):
            score = 0
            reasons = []
            
            if is_cw_end[i]:
                score += 10
                reasons.append("CW_END")
            
            if is_cw_start[i]:
                score += 1
                reasons.append("CW_START")
                
            candidates.append((score, i, reasons))

        # Sort by score (ascending)
        candidates.sort(key=lambda x: x[0])
        
        best_score, best_edge, best_reasons = candidates[0]
        
        # Format the "safety" string based on what it is NOT
        if best_score == 0:
            safety_str = "SAFE_BOTH"
        elif "CW_END" not in best_reasons:
            safety_str = "SAFE_CW_END" # It might be a start, but at least not an end
        elif "CW_START" not in best_reasons:
            safety_str = "SAFE_CW_START"
        else:
            safety_str = "UNSAFE" # Blocked both ways
            
        if debug:
            print(f"  [DEBUG] L vector: {L}")
            print(f"  [DEBUG] Candidates (Score: Edge [Reasons]):")
            for s, e, r in candidates:
                print(f"    {s}: Edge {e} {r}")
        
        return best_edge, safety_str

    def contract_vector(self, R: List[int], edge_idx: int) -> List[int]:
        n = len(R)
        new_R = []
        next_i = (edge_idx + 1) % n
        
        for k in range(n):
            if k == next_i:
                continue 
            
            if edge_idx >= k:
                dist = edge_idx - k
            else:
                dist = (n - k) + edge_idx
                
            val = R[k]
            if val > dist:
                new_val = val - 1
            else:
                new_val = val
            
            if k == edge_idx:
                new_val = max(1, R[edge_idx] - 1)
                
            new_R.append(new_val)
        return new_R

    def run_reduction(self):
        curr_R = self.original_R
        print(f"START: {curr_R}")
        
        step = 0
        while len(curr_R) > 3: 
            if all(x == curr_R[0] for x in curr_R):
                print(f"Reached CONSTANT state: {curr_R}")
                break
                
            print(f"\n--- Step {step} (n={len(curr_R)}) ---")
            edge, safe_type = self.find_reducible_edge(curr_R, debug=True)
            
            if edge is None:
                print("  Stuck!")
                break
            
            next_R = self.contract_vector(curr_R, edge)
            print(f"  Action: Contract edge {edge} ({safe_type}) -> {next_R}")
            
            # --- התיקון כאן: שמירה כמילון ולא כ-Tuple ---
            self.history.append({
                "step": step,
                "R_before": curr_R,
                "edge": edge,
                "type": safe_type,
                "R_after": next_R
            })
            
            curr_R = next_R
            step += 1
            
        print(f"\nDONE. Final R: {curr_R}")
        return curr_R

# --- בדיקה ---
# x = [1, 3, 4, 4, 4, 3, 3, 2]
# print("Testing User Case: ", x)
# planner = WheelReductionPlanner(x)
# planner.run_reduction()