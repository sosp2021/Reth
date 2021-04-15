from .a2c import A2CSolver
from .algorithm import Algorithm
from .ddpg import DDPGSolver
from .dqn import DQNSolver
from .pg import PGSolver
from .ppo import PPOSolver

SOLVER_DICT = {
    "a2c": A2CSolver,
    "ddpg": DDPGSolver,
    "dqn": DQNSolver,
    "pg": PGSolver,
    "ppo": PPOSolver,
}


def get_solver(name, **kwargs):
    if name in SOLVER_DICT:
        return SOLVER_DICT[name](**kwargs)
    else:
        raise Exception("Invalid solver name")
