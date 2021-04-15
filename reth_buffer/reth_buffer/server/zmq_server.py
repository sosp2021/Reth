import traceback

import zmq

from ..utils.shm_buffer import ShmBuffer


def start_zmq_server(port, buffer_service):
    shm_buffer_cache = {}
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    # zmq main loop
    while True:
        req = socket.recv_json()
        if "shm_buffer" in req:
            path = req["shm_buffer"]
            if path not in shm_buffer_cache:
                shm_buffer_cache[path] = ShmBuffer(path=path, mode="r+")
            shm_buffer = shm_buffer_cache[path]
            if "struct" in req:
                data = shm_buffer.read(req["struct"])
        msg_type = req.get("type")

        try:
            if msg_type == "echo":
                buffer = buffer_service.buffer
                socket.send_json(
                    {
                        "message": "ok",
                        "buffer": {
                            "batch_size": buffer_service.batch_size,
                            "root_dir": buffer.root_dir,
                            "capacity": buffer.capacity,
                            "struct": buffer.struct,
                            "sample_start": buffer_service.is_sample_started(),
                        },
                    }
                )
            elif msg_type == "append":
                *data, weights = data
                if weights.size == 0:
                    weights = None
                buffer_service.append(*data, weights=weights)
                socket.send_json({"message": "ok"})
            elif msg_type == "sample":
                if not buffer_service.is_sample_started():
                    socket.send_json({"error": "not enough data for sample"})
                    return
                size = req.get("size", 1)
                samples = buffer_service.batch_sample(size)
                flattened = []
                for indices, weights in samples:
                    flattened.append(indices)
                    flattened.append(weights)
                struct = shm_buffer.write(*flattened)
                socket.send_json(
                    {
                        "message": "ok",
                        "samplers": len(buffer_service.samplers),
                        "struct": struct,
                    }
                )
            elif msg_type == "update_priorities":
                indices, weights = data
                buffer_service.update_priorities(indices, weights)
                socket.send_json({"message": "ok"})
            else:
                socket.send_json({"error": f"invalid type {msg_type}"})
        except BaseException as e:
            traceback.print_exc()
            socket.send_json({"error": f"Server Error: {e}"})
