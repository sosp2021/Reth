import zmq

from ..utils.pack import deserialize, serialize


def sampler_loop(req_addr, sampler_info, idx):
    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)
    req_sock = ctx.socket(zmq.SUB)
    req_sock.subscribe(b"")
    req_sock.connect(req_addr)

    out_sock = ctx.socket(zmq.PUSH)
    out_sock.set_hwm(1)
    out_sock.bind(sampler_info["addrs"][idx])

    sampler = sampler_info["sampler_cls"](**sampler_info["kwargs"])

    sample_start = sampler_info["sample_start"]
    batch_size = sampler_info["batch_size"]
    cnt = 0

    try:
        while True:
            if sampler.ready_sample(batch_size) and cnt >= sample_start:
                r_list, w_list, _ = zmq.select([req_sock], [out_sock], [])
            else:
                r_list, w_list, _ = zmq.select([req_sock], [], [])
            if req_sock in r_list:
                while req_sock.poll(0) != 0:
                    req = req_sock.recv()
                    indices, weights, step_flag = deserialize(req)
                    if step_flag:
                        sampler.on_step()
                    sampler.update(indices, weights)
                    cnt += len(indices)
            if out_sock in w_list:
                indices, weights = sampler.sample(batch_size)
                out_sock.send(serialize([indices, weights]))
    except KeyboardInterrupt:
        # silent exit
        pass
