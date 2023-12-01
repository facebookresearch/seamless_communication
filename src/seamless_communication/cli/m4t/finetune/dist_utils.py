# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import logging
import os
from datetime import timedelta
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing

logger = logging.getLogger(__name__)


def is_dist_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank() -> int:
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    if not is_dist_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def get_world_size() -> int:
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(loggers: List[logging.Logger]) -> None:
    """Initializes the distributed backend"""
    torch.multiprocessing.set_start_method("spawn")
    if "RANK" not in os.environ:
        logger.error(
            "Cannot init disributed context, as environment varaibles are not set."
        )
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(
        f"Rank={rank} local rank={local_rank}, world_size={world_size}, is_master={rank == 0}"
    )
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=180),
    )
    logger.info(f"Setting cuda:{local_rank} as main device")
    if not is_main_process():
        for to_mute in loggers:
            to_mute.setLevel(logging.ERROR)
    torch.cuda.set_device(local_rank)
    dist.barrier()
