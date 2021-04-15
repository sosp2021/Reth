from aiohttp import web


def register_routes(app: web.Application):
    app.add_routes(
        [
            web.get("/echo", echo),
            web.get("/endpoints", endpoints),
        ]
    )


# GET /endpoints
async def endpoints(req):
    sio_ns = req.app["sio_ns"]
    return web.json_response(sio_ns.get_flattend_endpoints())


# GET /echo
async def echo(req):
    return web.json_response({**req.app["config"], "remote_ip": req.remote})
