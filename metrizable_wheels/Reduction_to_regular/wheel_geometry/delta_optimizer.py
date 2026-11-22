import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DeltaConstraint:
    u: int
    v: int
    rhs_expr: gp.LinExpr
    kind: str 
    origin_step: int 
    # לצורך דיבוג - נשמור את ההיסטוריה המילולית של האילוץ
    history_log: str = ""

class DeltaSystemSolver:
    def __init__(self, original_R: List[int], history: List[Dict]):
        self.original_R = original_R
        self.history = history
        self.constraints: List[DeltaConstraint] = []
        self.w_vars = {} 
        self.model = None

    def _get_contracted_index_map(self, n_current: int, edge_idx: int) -> List[int]:
        mapping = [-1] * n_current
        next_i = (edge_idx + 1) % n_current
        current_new_idx = 0
        new_idx_of_edge_node = -1
        
        for i in range(n_current):
            if i == next_i: continue 
            mapping[i] = current_new_idx
            if i == edge_idx: new_idx_of_edge_node = current_new_idx
            current_new_idx += 1
            
        mapping[next_i] = new_idx_of_edge_node
        return mapping

    def solve(self):
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            self.model = gp.Model(env=env)
        except:
            print("Gurobi initialization failed.")
            return

        # 1. Variables
        for step_idx, _ in enumerate(self.history):
            self.w_vars[step_idx] = self.model.addVar(lb=0.01, name=f"w_{step_idx}")

        self.model.update()
        
        print("\n=== DEBUGGING DELTA GENERATION ===")
        
        # active_constraints holds constraints from PREVIOUS steps
        active_constraints = []

        for step_idx, step_data in enumerate(self.history):
            R_curr = step_data['R_before']
            edge_idx = step_data['edge']
            n_curr = len(R_curr)
            w_k = self.w_vars[step_idx]
            
            print(f"\n--- Step {step_idx} (n={n_curr}): Contract Edge {edge_idx} ---")

            # --- A. TRANSLATE OLD CONSTRAINTS ---
            translated_constraints = []
            mapping = self._get_contracted_index_map(n_curr, edge_idx)
            print(f"  Mapping: {list(enumerate(mapping))}")
            
            for constr in active_constraints:
                u_curr, v_curr = constr.u, constr.v
                if u_curr == v_curr: continue 
                
                # Check crossing
                norm_edge = (edge_idx - u_curr) % n_curr
                norm_v = (v_curr - u_curr) % n_curr
                passes_edge = (norm_edge < norm_v)
                
                new_rhs = gp.LinExpr(constr.rhs_expr)
                added_w = False
                if passes_edge:
                    new_rhs.add(w_k, 1.0) 
                    added_w = True
                
                u_next = mapping[u_curr]
                v_next = mapping[v_curr]
                
                if u_next != v_next:
                    # Update Log
                    log_update = f" -> Map({u_curr},{v_curr})to({u_next},{v_next})"
                    if added_w: log_update += f"+w_{step_idx}"
                    
                    translated_constraints.append(DeltaConstraint(
                        u=u_next, v=v_next,
                        rhs_expr=new_rhs,
                        kind=constr.kind,
                        origin_step=constr.origin_step,
                        history_log=constr.history_log + log_update
                    ))
            
            # --- B. GENERATE NEW CONSTRAINTS ---
            new_constraints = []
            
            for u in range(n_curr):
                rim_reach = R_curr[u]
                if rim_reach == 0: continue
                
                v_pass = (u + rim_reach) % n_curr
                
                # 1. PASS CHECK
                norm_edge = (edge_idx - u) % n_curr
                norm_v = (v_pass - u) % n_curr
                
                if norm_edge < norm_v:
                    # Delta > w_k
                    pass_rhs = gp.LinExpr(0)
                    pass_rhs.add(w_k, 1.0)
                    
                    u_map, v_map = mapping[u], mapping[v_pass]
                    
                    print(f"  [GEN PASS] Path {u}->{v_pass} crosses edge {edge_idx}. "
                          f"Constraint: Delta({u_map},{v_map}) > w_{step_idx}")
                    
                    new_constraints.append(DeltaConstraint(
                        u=u_map, v=v_map,
                        rhs_expr=pass_rhs,
                        kind='PASS', origin_step=step_idx,
                        history_log=f"GenStep{step_idx}({u}->{v_pass})"
                    ))
                
                # 2. STOP CHECK
                v_stop = (v_pass + 1) % n_curr
                # Edge leading to v_stop is (v_stop-1)->v_stop
                edge_into_stop = (v_stop - 1) % n_curr
                
                if edge_into_stop == edge_idx:
                    # Delta < w_k
                    stop_rhs = gp.LinExpr(0)
                    stop_rhs.add(w_k, 1.0)
                    
                    u_map, v_map = mapping[u], mapping[v_stop]

                    print(f"  [GEN STOP] Path {u}->{v_stop} stops AT edge {edge_idx}. "
                          f"Constraint: Delta({u_map},{v_map}) < w_{step_idx}")

                    new_constraints.append(DeltaConstraint(
                        u=u_map, v=v_map,
                        rhs_expr=stop_rhs,
                        kind='STOP', origin_step=step_idx,
                        history_log=f"GenStep{step_idx}({u}->{v_stop})"
                    ))

            active_constraints = translated_constraints + new_constraints

        self._solve_deltas_only(active_constraints)

    def _solve_deltas_only(self, constraints: List[DeltaConstraint]):
        if not self.history: return
        final_R = self.history[-1]['R_after']
        n_base = len(final_R)
        
        print(f"\n--- Solving Abstract Delta System (n={n_base}) ---")
        
        delta_vars = {}
        for i in range(n_base):
            for j in range(i + 1, n_base):
                delta_vars[(i, j)] = self.model.addVar(lb=-GRB.INFINITY, name=f"Delta_{i}_{j}")

        eps = 1e-3
        
        print("\n--- Final Constraints with Trace ---")
        for i, c in enumerate(constraints):
            u, v = sorted((c.u, c.v))
            
            # Formatter
            rhs_str = ""
            for k in range(c.rhs_expr.size()):
                var = c.rhs_expr.getVar(k)
                coeff = c.rhs_expr.getCoeff(k)
                rhs_str += f"{coeff}*{var.VarName} + "
            rhs_str = rhs_str.rstrip(" + ") if rhs_str else "0"
            
            op = ">=" if c.kind == 'PASS' else "<="
            print(f"C{i} [{c.kind}]: Delta_{u}_{v} {op} {rhs_str}")
            print(f"    Trace: {c.history_log}")

            delta_var = delta_vars[(u, v)]
            
            if c.kind == 'PASS':
                self.model.addConstr(delta_var >= c.rhs_expr + eps)
            else: 
                self.model.addConstr(delta_var <= c.rhs_expr - eps)

        obj = gp.LinExpr()
        for k, var in self.w_vars.items():
            obj.add(var)
        self.model.setObjective(obj, GRB.MINIMIZE)
        
        self.model.optimize()
        
        if self.model.Status == GRB.OPTIMAL:
            print("\nSOLVED!")
            print("Base Deltas:")
            for (u, v), val in delta_vars.items():
                print(f"  Delta_{u}_{v}: {val.X:.4f}")
        
        elif self.model.Status == GRB.INFEASIBLE:
            print("\nINFEASIBLE")
            self.model.computeIIS()
            for constr in self.model.getConstrs():
                if constr.IISConstr:
                    print(f" -> {constr.ConstrName}")