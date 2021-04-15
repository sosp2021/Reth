import os.path as path

from reth.algorithm.util import calculate_discount_rewards
from reth.buffer import DynamicSizeBuffer
from reth.presets.config import get_solver, get_worker, get_trainer

MAX_TS = 100000
GAMMA = 0.99

if __name__ == "__main__":
    config_path = path.join(path.dirname(__file__), "config.yaml")
    solver = get_solver(config_path)
    # shared solver
    worker = get_worker(config_path, solver=solver)
    trainer = get_trainer(config_path, solver=solver)

    buffer = DynamicSizeBuffer(64)
    for _ in range(MAX_TS):
        for _ in range(10):
            s0, a, r, *_ = worker.step_episode()
            buffer.append_batch((s0, a, calculate_discount_rewards(r, GAMMA)))
        loss = trainer.step(buffer.data)
        buffer.clear()
