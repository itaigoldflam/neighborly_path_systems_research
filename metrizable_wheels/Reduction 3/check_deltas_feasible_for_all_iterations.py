import gurobipy as gp
from gurobipy import GRB
import random
import itertools
from tqdm import tqdm

def solve_system_for_specific_plan(num_vars, num_iterations, plan_indices, verbose=False):
    """
    Checks if a specific sequence of choices (plan) is feasible.
    
    Args:
        num_vars (int): Number of x variables.
        num_iterations (int): Number of iterations (t variables).
        plan_indices (list of lists): A list of length 'num_iterations'.
                                      Each element is a list of indices representing
                                      the 'Small' group for that iteration.
        verbose (bool): If True, prints detailed setup and solution values.
    
    Returns:
        bool: True if feasible, False otherwise.
    """
    
    # Create a new Gurobi model
    model = gp.Model("Iterative_System_Check")
    model.setParam('OutputFlag', 0)  # Mute solver output

    # --- Constants ---
    EPSILON = 1e-2 
    LARGE_BOUND = 10 

    # --- 1. Define Variables ---
    x_vars = model.addVars(num_vars, lb=EPSILON, ub=LARGE_BOUND, vtype=GRB.CONTINUOUS, name="x_init")
    t_vars = model.addVars(num_iterations, lb=EPSILON, ub=LARGE_BOUND, vtype=GRB.CONTINUOUS, name="t")

    # --- 2. Build Constraints Iteratively ---
    accumulated_subtractions = [0.0 for _ in range(num_vars)]

    if verbose:
        print(f"\n--- Building System for {num_vars} vars, {num_iterations} iters ---")

    for k in range(num_iterations):
        t_current = t_vars[k]
        
        # Identify Small and Large groups based on the plan
        small_indices = plan_indices[k]
        large_indices = [i for i in range(num_vars) if i not in small_indices]
        
        if verbose:
            print(f"Iter {k+1}: Small Indices={small_indices}, Large Indices={large_indices}")

        # Iterate over all variables to define logic and positivity
        for i in range(num_vars):
            # Expression for current value of x_i
            current_val_expr = x_vars[i] - accumulated_subtractions[i]
            
            # Constraint: Current value must be positive
            model.addConstr(current_val_expr >= EPSILON, name=f"Positivity_Iter{k}_Var{i}")

            if i in small_indices:
                # Logic: x_current < t_current
                model.addConstr(current_val_expr <= t_current - EPSILON, name=f"Logic_Small_Iter{k}_Var{i}")
            else:
                # Logic: x_current > t_current
                model.addConstr(current_val_expr >= t_current + EPSILON, name=f"Logic_Large_Iter{k}_Var{i}")

        # Update subtractions for the NEXT iteration
        for j in large_indices:
            accumulated_subtractions[j] += t_current
            
    # --- Final Positivity Check ---
    for i in range(num_vars):
        current_val_expr = x_vars[i] - accumulated_subtractions[i]
        model.addConstr(current_val_expr >= EPSILON, name=f"Positivity_Final_Var{i}")

    # --- 3. Optimize ---
    model.optimize()

    # --- 4. Print Results (If Verbose) ---
    if verbose:
        if model.status == GRB.OPTIMAL:
            print("\nSUCCESS: The system is FEASIBLE.")
            print("-" * 30)
            print("Initial X values found:")
            for i in range(num_vars):
                print(f"  x_{i+1} = {x_vars[i].X:.4f}")
            print("-" * 30)
            print("Threshold (t) values found:")
            for i in range(num_iterations):
                print(f"  t_{i+1} = {t_vars[i].X:.4f}")
        else:
            print("\nFAILURE: The system is INFEASIBLE.")

    return model.status == GRB.OPTIMAL

def generate_all_possible_plans(num_vars, num_iterations):
    """
    Generates every possible combination of 'Small' subsets for N iterations.
    Returns a list of all plans and the total count.
    """
    all_indices = list(range(num_vars))
    
    # 1. Generate all possible subsets for a single iteration
    possible_subsets = []
    for r in range(num_vars + 1):
        combinations = itertools.combinations(all_indices, r)
        possible_subsets.extend([list(c) for c in combinations])
    
    # 2. Cartesian product to get sequences of subsets for N iterations
    all_plans = list(itertools.product(possible_subsets, repeat=num_iterations))
    
    return all_plans, len(all_plans)

def main(num_vars, num_iters, run_exhaustive):
    """
    Main logic controller.
    """
    print(f"--- Configuration: Vars={num_vars}, Iters={num_iters}, Mode={'Exhaustive' if run_exhaustive else 'Random'} ---")

    if run_exhaustive:
        print("Generating all possible plans...")
        all_plans, total_count = generate_all_possible_plans(num_vars, num_iters)
        print(f"Total scenarios to check: {total_count}")
        
        fail_count = 0
        
        for idx, plan in tqdm(enumerate(all_plans), total=total_count):
            # Pass verbose=False to keep it silent unless it fails
            is_feasible = solve_system_for_specific_plan(num_vars, num_iters, plan, verbose=False)
            
            if not is_feasible:
                fail_count += 1
                print("\n" + "!" * 40)
                print(f"FAILED on plan: {plan}")
                print("!" * 40 + "\n")
                # Optional: break here if you want to stop on first error
                # return 
            
        print("-" * 30)
        if fail_count == 0:
            print(f"RESULT: FULL FREEDOM verified. All {total_count} scenarios are feasible.")
        else:
            print(f"RESULT: RESTRICTED FREEDOM. {fail_count} scenarios were infeasible.")

    else:
        # Random Mode
        random_plan = []
        for _ in range(num_iters):
            k_size = random.randint(0, num_vars)
            subset = random.sample(range(num_vars), k_size)
            random_plan.append(subset)
            
        # Pass verbose=True to see full details
        solve_system_for_specific_plan(num_vars, num_iters, random_plan, verbose=True)

if __name__ == "__main__":
    # --- User Arguments ---
    NUM_VARS = 5
    NUM_ITERS = 4
    
    # Mode Selection:
    # True = Check ALL possible subset combinations (Exhaustive).
    # False = Check one random combination (Detailed print).
    RUN_EXHAUSTIVE = True
    
    main(NUM_VARS, NUM_ITERS, RUN_EXHAUSTIVE)