import asyncio
import secrets

import reth
import perwez

ENV_NAME = "QbertNoFrameskip-v4"
WORKER_BATCH_SIZE = 64
WORKER_CNT = 8


async def _trainer(server_url):
    data_recv = perwez.AsyncRecvSocket(server_url, "data", broadcast=False)
    weight_send = perwez.AsyncSendSocket(server_url, "weight", broadcast=True)
    weight = secrets.token_bytes(50 * 1024 * 1024)

    cnt = 0
    while True:
        latest_res = await data_recv.recv()
        shapes = [x.shape for x in latest_res]
        cnt += 1
        if cnt % 5 == 0:
            print(f"recv data: {cnt}, shape: {shapes}")
            await weight_send.send(weight)
            await asyncio.sleep(0.1)


async def _worker(idx, server_url):
    weight_recv = perwez.AsyncRecvSocket(server_url, "weight", broadcast=True)
    data_send = perwez.AsyncSendSocket(server_url, "data", broadcast=False)

    env = reth.env.make(ENV_NAME)
    buffer = reth.buffer.NumpyBuffer(WORKER_BATCH_SIZE, circular=False)

    w_cnt = 0
    d_cnt = 0
    s0 = env.reset().astype("f4")
    while w_cnt < 10:
        if not await weight_recv.empty():
            await weight_recv.recv()
            w_cnt += 1
        s1, r, done, _ = env.step(env.action_space.sample())
        s1 = s1.astype("f4")
        buffer.append((s0, r, done, s1))

        if buffer.size == buffer.capacity:
            await data_send.send(buffer.data)
            d_cnt += 1
            buffer.clear()
            print(f"worker{idx}: recv weights {w_cnt}, send data {d_cnt}")

        if done:
            s0 = env.reset().astype("f4")
        else:
            s0 = s1


def test_pubsub_full():
    server_proc, config = perwez.start_server()
    loop = asyncio.get_event_loop()
    trainer_task = loop.create_task(_trainer(config["url"]))
    worker_tasks = []
    for idx in range(WORKER_CNT):
        wt = loop.create_task(_worker(idx, config["url"]))
        worker_tasks.append(wt)
    loop.run_until_complete(asyncio.gather(*worker_tasks))
    trainer_task.cancel()
    server_proc.terminate()
    server_proc.join()
