import ipaddress

from aiohttp import web


def register_routes(app: web.Application):
    app.add_routes(
        [
            web.get("/echo", echo),
            web.get("/endpoints", list_endpoints),
            web.post("/endpoints", add_endpoints),
            web.delete("/endpoints", delete_endpoints),
        ]
    )


# GET /echo
async def echo(req):
    remote_ip = req.remote
    if ipaddress.ip_address(remote_ip).is_loopback:
        remote_ip = req.app["config"]["ip"]
    return web.json_response({**req.app["config"], "remote_ip": remote_ip})


# GET /endpoints
async def list_endpoints(req):
    epm = req.app["endpoint_manager"]
    return web.json_response(epm.query())


# POST /endpoints
async def add_endpoints(req):
    epm = req.app["endpoint_manager"]
    epm.add(await req.json())
    return web.json_response({"msg": "ok"})


# DELETE /endpoints
async def delete_endpoints(req):
    epm = req.app["endpoint_manager"]
    payload = await req.json()
    epm.remove(payload["addr"])
    return web.json_response({"msg": "ok"})
