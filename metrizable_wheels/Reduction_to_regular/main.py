# main.py
from wheel_geometry.reduction import WheelReductionPlanner
from wheel_geometry.delta_optimizer import DeltaSystemSolver

R_init = [3, 3, 2, 2, 1, 2]

# 1. Run Reduction
planner = WheelReductionPlanner(R_init)
final_R = planner.run_reduction()

# 2. Run Optimizer
solver = DeltaSystemSolver(R_init, planner.history)
solver.solve()