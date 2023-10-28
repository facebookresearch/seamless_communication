# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import List

import torch
import yaml
from m4t_scripts.train import dataloader as _dataloader
from m4t_scripts.train import dist_utils
from m4t_scripts.train import model as _model
from m4t_scripts.train import trainer as _trainer
from m4t_scripts.train.configs import WorkflowParams

logging_format = f"%(asctime)s - {platform.node()} - %(process)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
)

logger = logging.getLogger("train")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M4T training")
    parser.add_argument(
        "--wd",
        type=Path,
        required=True,
        help="Work directory, where logs, checkpoints and core dumps will be stored",
    )
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        help="Config with training parameters",
    )
    return parser


def run_training(
    parameters: WorkflowParams, work_dir: str, checkpoint_dir: str
) -> None:
    logger.info(f"Workflow params: {parameters}")
    rank, world_size = dist_utils.get_rank(), dist_utils.get_world_size()
    logger.info(f"Rank: {rank}, world_size: {world_size}")
    assert torch.cuda.device_count() > 0, "GPU is not available"
    device = torch.device("cuda")
    float_dtype = _trainer.UnitYTrainer._get_float_dtype(
        parameters.training.float_dtype
    )
    logger.info(f"Device: {device}, float dtype: {float_dtype}")
    model = _model.ModelBuilder(
        config=parameters.model, dtype=float_dtype, device=device
    ).build_model()
    logger.info(f"Model: {model}")
    train_data = _dataloader.UnityDataLoader(
        config=parameters.train_data,
        rank=rank,
        world_size=world_size,
        target_device=device,
        float_dtype=float_dtype,
    )
    eval_data = _dataloader.UnityDataLoader(
        config=parameters.eval_data,
        rank=rank,
        world_size=world_size,
        target_device=device,
        float_dtype=float_dtype,
    )
    trainer = _trainer.UnitYTrainer(
        model=model,
        params=parameters.training,
        train_data_loader=train_data,
        eval_data_loader=eval_data,
        chck_save_dir=checkpoint_dir,
        device=device,
    )
    trainer.run()


def get_loggers() -> List[logging.Logger]:
    return [
        logger,
        _trainer.logger,
        _dataloader.logger,
        _model.logger,
        dist_utils.logger,
    ]


def set_file_output_for_loggers(log_filename: str) -> None:
    handler = logging.FileHandler(filename=log_filename, mode="a", delay=False)
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)
    for logger in get_loggers():
        logger.handlers.append(handler)


def main() -> None:
    args = init_parser().parse_args()
    dist_utils.init_distributed(get_loggers())
    is_master = dist_utils.is_main_process()
    with open(args.params, "r") as fp_in:
        parameters = WorkflowParams.deserialize(
            yaml.load(fp_in, Loader=yaml.FullLoader)
        )
    ts = str(int(time.time()))
    work_dir = args.wd
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir) and is_master:
        logger.info(f"Creating checkpoint dir: {checkpoint_dir}")
        # checkpoint_dir is not going to be used before syncs downstream,
        #   so don't expect racing condition, and don't run barrier
        os.makedirs(checkpoint_dir)
    config_path = os.path.join(work_dir, f"{ts}_config.yaml")
    # copy to work dir to keep a snapshot of workflow config
    if is_master:
        shutil.copy(args.params, config_path)
    log_path = os.path.join(work_dir, "train_log.txt")
    logger.info(f"Set logging to {log_path}")
    set_file_output_for_loggers(log_path)
    try:
        run_training(
            parameters=parameters, work_dir=work_dir, checkpoint_dir=checkpoint_dir
        )
    except Exception:
        # make sure that the stack tracke will be logged to log files
        logger.exception("Training failed")


if __name__ == "__main__":
    main()
