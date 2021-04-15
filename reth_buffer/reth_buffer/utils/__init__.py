import os
import socket
from uuid import uuid4

import numpy as np

ROOT_DIR_PREFIX = "/dev/shm/reth_buffer_"
QUEUE_SIZE_PER_THREAD = 10
