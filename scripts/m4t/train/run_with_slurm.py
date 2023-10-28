import argparse
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path


logging_format = f"%(asctime)s - {platform.node()} - %(process)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
)

logger = logging.getLogger("train")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M4T training")
    parser.add_argument(
        "-w",
        type=Path,
        required=True,
        help="Work directory, where logs, checkpoints and core dumps will be stored",
    )
    parser.add_argument(
        "-p",
        type=Path,
        required=True,
        help="Training workflow config",
    )
    parser.add_argument(
        "-n",
        type=int,
        required=False,
        default=1,
        help="Number of training nodes",
    )
    parser.add_argument(
        "-c",
        type=str,
        required=False,
        default="seamless",
        help="Cluster partitions to use",
    )
    parser.add_argument(
        "-j",
        type=str,
        required=False,
        default="train",
        help="Slurm job name",
    )
    return parser


def prepare_sbatch_config(
    job_name: str,
    params_file: str,
    num_nodes: int,
    partitions: str,
    work_dir: str,
    cluster_logs_dir: str,
    run_script: str,
) -> str:
    return f"""#!/bin/bash
## job name
#SBATCH --job-name={job_name}

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output={cluster_logs_dir}/%j.out

## filename for job standard error output (stderr)
#SBATCH --error={cluster_logs_dir}/%j.err

## partition name
#SBATCH --partition={partitions}

## number of nodes
#SBATCH --nodes={num_nodes}

## number of nodes
#SBATCH --gpus-per-node=8

## number of cpus per task
#SBATCH --cpus-per-task=96

#SBATCH --gres=gpu:8

## number of tasks per node
#SBATCH --ntasks-per-node=1

## amount of mem
#SBATCH --mem 500G

## amount of time in minutes
#SBATCH --time 2400

set -x
export WANDB_DISABLED=true
export HDF5_USE_FILE_LOCKING='FALSE'
export PARENT=`/bin/hostname -s`
export MPORT=24198
export CHILDREN=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT`
export HOSTLIST="$PARENT $CHILDREN"
echo $HOSTLIST
export WORLD_SIZE=$SLURM_NTASKS
srun --label bash -c 'which python && torchrun \\
 --nproc_per_node=8 \\
 --nnodes=$SLURM_JOB_NUM_NODES \\
 --node_rank="$SLURM_PROCID" \\
 --master_addr="$PARENT" \\
 --master_port="$MPORT" \\
 --log-dir={cluster_logs_dir} \\
{run_script} --params {params_file}  --wd {work_dir}'
"""


def main() -> None:
    args = init_parser().parse_args()
    params_file = args.p
    num_nodes = args.n
    partitions = args.c
    work_dir = args.w
    job_name = args.j

    assert job_name is not None
    assert len(job_name.split()) == 1, "spaces in job name not allowed"
    assert (
        partitions and len(partitions.split()) == 1
    ), "spaces in partitions not allowed"
    assert os.path.exists(params_file), "config file is missing"
    training_script_path = os.path.join(os.path.dirname(__file__), "run_training.py")
    assert os.path.exists(
        training_script_path
    ), f"Can't find training script {training_script_path}"
    assert num_nodes > 0
    if not os.path.exists(work_dir):
        logger.info(f"Creating workdir {work_dir}")
        os.makedirs(work_dir)
    cluster_logs_dir = os.path.join(work_dir, "cluster_logs")
    if os.path.exists(cluster_logs_dir):
        logger.info(f"Clearing cluster logs dir {cluster_logs_dir}")
        shutil.rmtree(cluster_logs_dir)
    os.makedirs(cluster_logs_dir)
    config_text = prepare_sbatch_config(
        job_name=job_name,
        params_file=params_file,
        num_nodes=num_nodes,
        partitions=partitions,
        work_dir=work_dir,
        cluster_logs_dir=cluster_logs_dir,
        run_script=training_script_path,
    )
    logger.info(f"SBATCH config to launch: \n{config_text}")
    fname = f"{int(time.time())}_sbatch.sh"
    config_path = os.path.join(work_dir, fname)
    with open(config_path, "w") as fp_out:
        fp_out.write(config_text)
        logger.info(f"Saved to {config_path}")
    command = f"sbatch {config_path}"
    logger.info(f"Executing command: '{command}'")
    subprocess.Popen(command, shell=True).communicate()
    logger.info(f"Train log: {os.path.join(work_dir, 'train_log.txt')}")


if __name__ == "__main__":
    main()
