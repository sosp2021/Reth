import zmq


def get_zmq_socket(port):
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.IMMEDIATE, True)
    socket.connect(f"tcp://localhost:{port}")
    return socket
