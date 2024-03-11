#  Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import logging
import os
from datetime import timedelta
from pprint import pformat
from typing import Any, Optional

import submitit
import torch
import torch.distributed as dist
from fairseq2.gang import (
    Gang,
    ProcessGroupGang,
    _get_device_index,
    _get_num_cpus,
    _get_num_processes,
)
from fairseq2.typing import CPU, Device
from fairseq2.utils.profiler import log_environment_info
from fairseq2.utils.version import _is_pt22_or_greater

logger = logging.getLogger(__name__)


def _determine_default_device() -> Device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return _determine_default_cuda_device()

    return CPU


def _determine_default_cuda_device() -> Device:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")

    if visible_devices is not None:
        try:
            int(visible_devices)

        except ValueError:
            # If we are here, it means CUDA_VISIBLE_DEVICES is a list instead of
            # a single device index.
            device = None
        else:
            # By default in FS2, index=0 assuming that only the right device is exposed.
            device = Device("cuda", index=int(visible_devices))
    else:
        device = None

    if device is None:
        num_devices = torch.cuda.device_count()

        idx = _get_device_index(num_devices, device_name="CUDA")

        device = Device("cuda", index=idx)

    # As of PyTorch 2.0, FSDP fails to work if the default device is not set.
    torch.cuda.set_device(device)

    return device


class SubmititProcessGroupGang(ProcessGroupGang):  # type: ignore
    """Temporary solution to how we initialize our distibuted env with hydra+submitit"""

    @staticmethod
    def init_default_process_group(
        *,
        device: Optional[Device] = None,
        timeout: Optional[timedelta] = None,
        num_threads: Optional[int] = None,
        warn_only: bool = False,
        ok_initialized: bool = False,
    ) -> Gang:
        """Initialize the default process group and wrap it as a gang.
        this is exactly the same thing as  ProcessGroupGang.init_default_process_group
        except from  how we determine the CUDA device per worker in
        _determine_default_cuda_device
        """
        if not dist.is_available():
            raise RuntimeError("`torch.distributed` is not available.")

        if dist.is_initialized():
            if ok_initialized:
                return ProcessGroupGang.from_default_process_group()

            raise RuntimeError("The default process group is already initialized.")

        num_procs = _get_num_processes()

        if num_threads is None:
            if num_procs > 1 and "OMP_NUM_THREADS" not in os.environ:
                # To prevent thread oversubscription, we distribute cores evenly
                # across workers.
                num_threads = _get_num_cpus(num_procs)

        if num_threads is not None:
            torch.set_num_threads(num_threads)
            logger.info("Setting the number of threads used for intraop parallelism to %d.", num_threads)  # fmt: skip

        if device is None:
            device = _determine_default_device()

            assert device.type == "cpu" or device.type == "cuda"

        if device.type == "cpu":
            backend = "gloo"
        elif device.type == "cuda":
            backend = "nccl"
        else:
            raise ValueError(
                f"`device` must be of type 'cpu' and 'cuda', but is of type '{device.type}' instead."
            )

        if device.type == "cuda":

            def check_async_handling() -> None:
                env_name = "NCCL_ASYNC_ERROR_HANDLING"
                if env_name in os.environ:
                    return

                if _is_pt22_or_greater():
                    env_name = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
                    if env_name in os.environ:
                        return

                if warn_only:
                    logger.warning("The default process group uses the NCCL backend, but the `%s` environment variable is not set. \
                                   Your collective communication calls can hang indefinitely. \
                                   Learn more at https://github.com/pytorch/pytorch/issues/46874.", env_name)  # fmt: skip
                else:
                    raise RuntimeError(
                        f"The default process group uses the NCCL backend, but the `{env_name}` environment variable is not set. \
                        Learn more at https://github.com/pytorch/pytorch/issues/46874."
                    )

            check_async_handling()

        if timeout is None:
            timeout = timedelta(minutes=15)

        dist.init_process_group(backend, timeout=timeout)

        if dist.group.WORLD is None:
            raise RuntimeError(
                "The default process group is not available. Please file a bug report."
            )

        return ProcessGroupGang(dist.group.WORLD, device)


def print_env():
    for key in sorted(os.environ.keys()):
        if not (
            key.startswith(("SLURM_", "SUBMITIT_"))
            or key
            in (
                "MASTER_ADDR",
                "MASTER_PORT",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
            )
        ):
            continue
        value = os.environ[key]
        logger.info(f"R{dist.get_rank()} -- {key}={value}")


def init_process_group(config: Any, logger: logging.Logger) -> Gang:

    if getattr(config, "use_submitit", True):
        try:
            dist_env = submitit.helpers.TorchDistributedEnvironment().export(
                overwrite=True
            )
            # missing from submitit env
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

            assert dist_env.rank == dist.get_rank()
            assert dist_env.world_size == dist.get_world_size()

        except RuntimeError:
            import warnings

            warnings.warn(
                "looks like you are not in a submitit/stopes job. You probably want to override use_submitit=false"
            )

    gang = SubmititProcessGroupGang.init_default_process_group()
    print_env()

    if gang.rank == 0:
        logger.info("Job Config\n%s", pformat(config))

    log_environment_info(logger, gang.device)

    return gang


def set_mkl_num_threads():
    """Setting mkl num threads to 1, so that we don't get thread explosion."""
    mkl_rt = ctypes.CDLL("libmkl_rt.so")
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
