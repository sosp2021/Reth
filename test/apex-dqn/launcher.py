import asyncio
import os

import asyncssh
import portpicker
from perwez.utils import get_local_ip

TRAINER_ROLE_NAME = "trainer"
WORKER_ROLE_NAME = "worker"
WORKER_CNT = int(os.environ.get(f"PAI_TASK_ROLE_TASK_COUNT_{WORKER_ROLE_NAME}"))

TRAINER_ENDPOINT = f"{TRAINER_ROLE_NAME}-0"
WORKER_ENDPOINTS = [f"{WORKER_ROLE_NAME}-{i}" for i in range(WORKER_CNT)]

WORKER_COMMAND = "python -u ~/reth/test/apex-dqn/worker.py -r {} -s {} -b {}"

TRAINER_COMMAND = "python -u ~/reth/test/apex-dqn/trainer.py -p {}"

QUEUE = asyncio.Queue()


async def printer():
    while True:
        line = await QUEUE.get()
        print(line, end="")


async def handle_output(prefix, stream):
    async for line in stream:
        if len(line) == 0:
            continue
        await QUEUE.put(f"{prefix} {line}")


async def run(name, ep, command):
    async with asyncssh.connect(
        ep, known_hosts=None, config="/etc/ssh/ssh_config"
    ) as conn:
        async with conn.create_process(command) as proc:
            await asyncio.gather(
                handle_output(f"[{name}][stdout]", proc.stdout),
                handle_output(f"[{name}][stderr]", proc.stderr),
            )


async def main_async():
    port = portpicker.pick_unused_port()
    local_ip = get_local_ip()
    local_addr = f"http://{local_ip}:{port}"

    tasks = []
    tasks.append(run("trainer", TRAINER_ENDPOINT, TRAINER_COMMAND.format(port)))
    for i, ep in enumerate(WORKER_ENDPOINTS):
        tasks.append(
            run(
                f"worker-node{i}",
                ep,
                WORKER_COMMAND.format(i, len(WORKER_ENDPOINTS), local_addr),
            )
        )
    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for fut in done:
        fut.result()


def main():
    loop = asyncio.get_event_loop()
    p_task = loop.create_task(printer())
    loop.run_until_complete(main_async())
    p_task.cancel()


if __name__ == "__main__":
    main()
