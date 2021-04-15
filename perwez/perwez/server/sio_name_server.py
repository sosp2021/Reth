import asyncio

import socketio


class SioNameServer:
    def __init__(self, sio_server: socketio.AsyncServer, parent_url=None):
        self.loop = asyncio.get_event_loop()

        self.sio_server = sio_server
        self.sio_server.on("connect", self._on_connect)
        self.sio_server.on("disconnect", self._on_disconnect)
        self.sio_server.on("register", self._on_register)

        self.parent_sio = None
        if parent_url is not None:
            self.parent_sio = socketio.AsyncClient()
            self.parent_sio.on("connect", self._data_up)
            self.parent_sio.on("update", self._on_parent_updated)
            self.loop.create_task(
                self.parent_sio.connect(parent_url, transports="websocket")
            )

        self.global_endpoints = []
        # sid -> endpoints
        self.local_tcp_endpoints = {}
        self.local_ipc_endpoints = {}

    def get_flattend_endpoints(self):
        flattend = []
        if self.parent_sio is None:
            for val in self.local_tcp_endpoints.values():
                flattend.extend(val)
        else:
            flattend.extend(self.global_endpoints)

        for val in self.local_ipc_endpoints.values():
            flattend.extend(val)
        return flattend

    async def _data_up(self):
        assert self.parent_sio is not None
        # tcp only
        flattend = []
        for val in self.local_tcp_endpoints.values():
            flattend.extend(val)
        await self.parent_sio.emit("register", flattend)

    async def _data_down(self, to=None):
        await self.sio_server.emit("update", self.get_flattend_endpoints(), to=to)

    async def _on_changed(self):
        if self.parent_sio is None:
            await self._data_down()
        else:
            await self._data_up()

    async def _on_connect(self, sid, _):
        await self._data_down(to=sid)

    async def _on_disconnect(self, sid):
        flag_changed = False
        if sid in self.local_ipc_endpoints:
            flag_changed = True
            del self.local_ipc_endpoints[sid]
        if sid in self.local_tcp_endpoints:
            flag_changed = True
            del self.local_tcp_endpoints[sid]
        if flag_changed:
            await self._on_changed()

    async def _on_register(self, sid, endpoints):
        if len(endpoints) == 0:
            return
        ipc = [x for x in endpoints if x["addr"].startswith("ipc")]
        tcp = [x for x in endpoints if x["addr"].startswith("tcp")]
        self.local_ipc_endpoints[sid] = ipc
        self.local_tcp_endpoints[sid] = tcp
        await self._on_changed()

    async def _on_parent_updated(self, endpoints):
        self.global_endpoints = [x for x in endpoints if x["addr"].startswith("tcp")]
        await self._data_down()
