import asyncio

import asyncssh

WORKER_ENDPOINTS = ["10.0.0.1:234"]
TRAINER_ENDPOINT = "10.0.0.2:345"
PORT = 2333


WORKER_COMMAND = "python ~/reth/test/apex-dqn/worker.py -r {} -s {} -b {}"

TRAINER_COMMAND = "python ~/reth/test/apex-dqn/trainer.py -p {}"

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


def parse_endpoint(ep):
    if ":" not in ep:
        host = ep
        port = 22
    else:
        host, port = ep.split(":")
    if "@" not in host:
        user = "root"
        addr = host
    else:
        user, addr = host.split("@")
    return user, addr, port


async def run(name, ep, command):
    user, addr, port = parse_endpoint(ep)
    async with asyncssh.connect(
        addr, username=user, port=int(port), known_hosts=None
    ) as conn:
        async with conn.create_process(command) as proc:
            await asyncio.gather(
                handle_output(f"[{name}][stdout]", proc.stdout),
                handle_output(f"[{name}][stderr]", proc.stderr),
            )


async def main_async():
    _, addr, _ = parse_endpoint(TRAINER_ENDPOINT)
    local_addr = f"http://{addr}:{PORT}"
    tasks = []
    tasks.append(run("trainer", TRAINER_ENDPOINT, TRAINER_COMMAND.format(PORT)))
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
