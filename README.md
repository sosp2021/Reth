# Project Reth

## Packages

- perwez: cross-node pubsub communication
- reth: basic reinforcement learning algorithm / utilities
- reth_buffer: intra-node shared-mem based prioritized replay buffer service

## Installation

The project's dependencies is curretly managed with [poetry](https://python-poetry.org/docs/) package manager.

### Quick Start Example

```bash
## install poetry
pip install poetry
## disable poetry's virtualenvs feature
poetry config virtualenvs.create false
## use the dev pyproject.toml file under the git directory for installation
## it will install all 3 packages in develop mode (like pip install -e)
poetry install
```

### Extra Dependencies

- `torch` for `reth`'s algorithms
- `opencv-python` for `reth`'s atari env wrappers
- `horovod` for apex-dqn horovod example
- `reverb` for apex-dqn reverb example
