import os.path as path

from reth.buffer import PrioritizedBuffer
from reth.presets.config import get_solver, get_worker, get_trainer, get_replay_buffer

BATCH_SIZE = 64
MAX_TS = 100000

if __name__ == "__main__":
    config_path = path.join(path.dirname(__file__), "config.yaml")
    solver = get_solver(config_path)
    # shared solver
    worker = get_worker(config_path, solver=solver)
    trainer = get_trainer(config_path, solver=solver)
    buffer = get_replay_buffer(config_path)
    assert isinstance(buffer, PrioritizedBuffer)

    # init buffer
    init_data = worker.step_batch(1000)
    buffer.append_batch(init_data)

    for _ in range(MAX_TS):
        # worker
        data = worker.step_batch(BATCH_SIZE)
        loss = solver.calc_loss(data)
        buffer.append_batch(data)
        # trainer
        data, indices, weights = buffer.sample(BATCH_SIZE)
        loss = trainer.step(data)
        buffer.update_priorities(indices, loss)
