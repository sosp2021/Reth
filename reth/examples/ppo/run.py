import os.path as path

from reth.algorithm.util import calculate_discount_rewards_with_dones
from reth.buffer import DynamicSizeBuffer
from reth.presets.config import get_solver, get_worker, get_trainer

MAX_TS = 100000
GAMMA = 0.99
UPDATE_INTERVAL = 2000

if __name__ == "__main__":
    config_path = path.join(path.dirname(__file__), "config.yaml")
    solver = get_solver(config_path)
    act_solver = get_solver(config_path)
    # shared solver
    worker = get_worker(config_path, solver=act_solver)
    trainer = get_trainer(config_path, solver=solver)

    episode_buffer = DynamicSizeBuffer(64)
    data_buffer = DynamicSizeBuffer(64)
    act_solver.sync_weights(solver)
    for _ in range(MAX_TS):
        for _ in range(UPDATE_INTERVAL):
            a, logprob = act_solver.act(worker.s0)
            s0, a, r, s1, done = worker.step(a)
            episode_buffer.append((s0, a, r, logprob, done))
            if done:
                s0, a, r, logprob, done = episode_buffer.data
                r = calculate_discount_rewards_with_dones(r, done, GAMMA)
                data_buffer.append_batch((s0, a, r, logprob))
                episode_buffer.clear()

        loss = trainer.step(data_buffer.data)
        act_solver.sync_weights(solver)
        data_buffer.clear()
