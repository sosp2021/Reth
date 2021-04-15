import multiprocessing as mp
import time

import numpy as np

BATCH_SIZE = 64
CAPACITY = 10000
TEST_CNT = 1000


def test_reth_buffer(data):
    import reth_buffer

    print("TEST RETH_BUFFER")
    print("initializing...")
    buffer_process, addr = reth_buffer.start_server(CAPACITY, BATCH_SIZE)
    client = reth_buffer.Client(addr)
    client.append(data, np.ones(CAPACITY))
    # loader = reth_buffer.TorchCudaLoader(addr)
    loader = reth_buffer.NumpyLoader(addr)
    # init
    loader.sample()
    print("ready")
    t0 = time.perf_counter()
    for _ in range(TEST_CNT):
        loader.sample()
    t1 = time.perf_counter()
    print(TEST_CNT, t1 - t0)
    buffer_process.terminate()
    buffer_process.join()


def test_reverb(data):
    import reverb
    import tensorflow as tf

    print("TEST REVERB")
    print("initializing...")
    reverb_server = reverb.Server(
        tables=[
            reverb.Table(
                name="req",
                sampler=reverb.selectors.Prioritized(0.6),
                remover=reverb.selectors.Fifo(),
                max_size=CAPACITY,
                rate_limiter=reverb.rate_limiters.MinSize(100),
            )
        ],
        port=15867,
    )
    client = reverb_server.in_process_client()
    for i in range(CAPACITY):
        client.insert([col[i] for col in data], {"req": np.random.rand()})
    dataset = reverb.ReplayDataset(
        server_address="localhost:15867",
        table="req",
        dtypes=(tf.float64, tf.float64, tf.float64, tf.float64),
        shapes=(
            tf.TensorShape([1, 84, 84]),
            tf.TensorShape([1, 84, 84]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),
        max_in_flight_samples_per_worker=10,
    )
    dataset = dataset.batch(BATCH_SIZE)
    print("ready")
    t0 = time.perf_counter()
    for sample in dataset.take(TEST_CNT):
        pass
    t1 = time.perf_counter()
    print(TEST_CNT, t1 - t0)


def main():
    print("init data...")
    data = []
    data.extend(np.random.rand(CAPACITY, 1, 84, 84) * 20 for _ in range(2))
    data.extend(np.random.rand(CAPACITY) * 20 for _ in range(2))

    test_reth_buffer(data)

    # test_reverb(data)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
