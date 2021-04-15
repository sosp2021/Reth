import io
import os

import yaml

import reth

from .trainer import Trainer
from .worker import Worker
from ..buffer import NumpyBuffer, PrioritizedBuffer


def _parse_input(f):
    res = None
    if isinstance(f, str):
        if os.path.exists(f):
            with open(f, "r") as yaml_file:
                res = yaml.safe_load(yaml_file)
        else:
            res = yaml.safe_load(f)
    elif isinstance(f, io.IOBase):
        res = yaml.safe_load(f)
    elif isinstance(f, dict):
        res = f
    else:
        raise Exception("Invalid config input", f)

    return res


def get_env(f, **kwargs):
    config = _parse_input(f)
    return reth.env.make(**{**config["env"], **kwargs})


def get_solver(f, env=None, **kwargs):
    config = _parse_input(f)
    if env is None:
        env = get_env(config)
    return reth.algorithm.get_solver(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **{**config["solver"], **kwargs}
    )


def get_worker(f, solver=None, env=None, **kwargs):
    config = _parse_input(f)
    if env is None:
        env = get_env(config)
    if solver is None:
        solver = get_solver(config, env)
    return Worker(env, solver, **{**config["worker"], **kwargs})


def get_trainer(f, solver=None, env=None, **kwargs):
    config = _parse_input(f)
    if env is None:
        env = get_env(config)
    if solver is None:
        solver = get_solver(config, env)
    return Trainer(solver, **{**config["trainer"], **kwargs})


def get_replay_buffer(f, **kwargs):
    config = _parse_input(f)
    buffer_config = config["replay_buffer"]
    prioritized = buffer_config["prioritized"]
    config_args = {k: buffer_config[k] for k in buffer_config if k != "prioritized"}
    if prioritized:
        return PrioritizedBuffer(**{**config_args, **kwargs})
    else:
        return NumpyBuffer(**{**config_args, **kwargs})
