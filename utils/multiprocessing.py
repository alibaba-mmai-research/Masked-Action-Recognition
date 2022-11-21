#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Task launcher. """

import os, sys, time
import torch
import datetime
def launch_task(cfg, init_method, func):
    """
    Launches the task "func" on one or multiple devices.
    Args:
        cfg (Config): global config object. 
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): task to run.
    """
    torch.cuda.empty_cache()
    os.environ['NCCL_BLOCKING_WAIT'] = "1"
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = "0"
    if cfg.NUM_GPUS > 1 or cfg.NUM_SHARDS > 1:
        if cfg.PAI or cfg.JIUDING:
            cfg.SHARD_ID = int(os.environ['RANK'])
            if "VISIBLE_DEVICE_LIST" in os.environ:
                cfg.NUM_GPUS = len(os.environ["VISIBLE_DEVICE_LIST"].split(","))
            else:
                cfg.NUM_GPUS = torch.cuda.device_count()
            cfg.NUM_SHARDS = int(os.environ['WORLD_SIZE'])

        torch.multiprocessing.spawn(
            run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        cfg.LOCAL_RANK = 0
        func(cfg=cfg)

def run(
    local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (Config): global config object.
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank
    cfg.LOCAL_RANK = local_rank

    print("num_proc (NUM_GPU): {}".format(num_proc))
    print("shard_id (os.environ['RANK']): {}".format(shard_id))
    print("num_shards (os.environ['WORLD_SIZE']): {}".format(num_shards))
    print("rank: {}".format(rank))
    print("local_rank (GPU_ID): {}".format(local_rank))
    sys.stdout.flush()
    if "VISIBLE_DEVICE_LIST" in os.environ:
        torch.cuda.set_device(int(os.environ["VISIBLE_DEVICE_LIST"]))
    else:
        torch.cuda.set_device(f'cuda:{local_rank}')
    if rank != 0 and '1.7' in torch.__version__:
        time.sleep(60)
    try:
        if cfg.PAI == False:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        else:
            torch.distributed.init_process_group(
                backend=backend,
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(seconds=36000),
            )
    except Exception as e:
        raise e
    
    # if cfg.MULTI_CARD is False:
    #     os.system(f"CUDA_VISIBLE_DEVICES={local_rank}")
    func(cfg)
