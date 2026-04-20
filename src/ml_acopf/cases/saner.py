import copy

from ml_acopf.cases.networks import build_network
from ml_acopf.config import load_config
from ml_acopf.solver.solver import solve_ac_opf

cfg = load_config("configs/case118.toml")
base_net = build_network("case118")

for i in range(10):
    stats = solve_ac_opf(copy.deepcopy(base_net), cfg.solver)
    print(i, stats.success, stats.termination_status, stats.error)
