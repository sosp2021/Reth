# Perwez

## Quick Start

### init

```python
import perwez
server_proc, port = perwez.start_server(name='default')
client = perwez.connect(name='default')
```

### publish

```python
client.publish('topic', 'data')
```

### subscribe

```python
watcher = client.subscribe('topic')

data = watcher.get(block=True)

or

bytes_io = watcher.stream()
```

## API

### `perwez.start_server(name, host='0.0.0.0', port=None, parent_url=None, lmdb_dir=None, lmdb_args={})`

When `perwez.init()` is called, a local http server with specified name will be started as a subprocess if not exists.

The `multiprocessing.Process` object and server's port number will be returned

#### name

type: str

The local http server's identifier. Exception will be raised if a local server with the specified name is already started.

#### host

type: str

The host address that the newly created local server will be listened to.

#### port

type: int

The port number that the newly created local server will be listened to.

Will be chose automatically if not specified.

#### parent_url

type: str

The parent url that the newly created local server will be connected to.

The cluster's pub/sub info will be synced through connected server.

### `perwez.connect(name)`

Connect to the exists local server with specified name and return a `perwez.Client` object

### `perwez.Client(url)`

Create a client object connected to specified perwez server

#### url

type: str

server's url

### `client.publish(topic, data, slot=0)`

Publish data to the network.

#### topic

type: str

#### data

type: any

Data that will be published. Non binary data will be pickled automatically.

#### slot

type: number

Data's extra identifier. Previous published data with same topic and slot will be overwritten.

### `client.subscribe(topic, downloaded_handler=None)`

Subscribe a topic and return a `perwez.client.Watcher` object.

Data with specified topic will be downloaded by local http server and pushed to the returned `perwez.client.Watcher`'s data queue.

#### topic

type: str

#### downloaded_handler

type: func

Please refer to `watcher.set_downloaded_handler()`

### `watcher.empty()`

Return True if the watcher's queue is empty.

### `watcher.get(block=True)`

Remove an item from the queue and return its data.

Data will be deserialized if it is pickled.

#### block

type: bool

If block is set, the function will wait for a new item when queue is empty.

Otherwise, a `Queue.Empty` exception will be raised.

### `watcher.get_event(block=True)`

Remove an item from the queue and return the tuple of event object and raw data byte array.

### `watcher.stream(block=True)`

Remove an item from the queue and return its raw data as a `BytesIO` object.

### `watcher.set_downloaded_handler(handler)`

bytes data and event's payload will be passed to the custom handler.

If the custom handler returns `True`, the event will be pushed into watcher's queue. Otherwise, the event is treated as processed and will be dropped.

e.g.

```python
def handler(data, event):
    print('Bytes data', data)
    print('Event info', event)

    return False # The event won't be pushed into watcher's queue.
```

## Local server API

TODO
