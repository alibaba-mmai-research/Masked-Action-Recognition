#!/usr/bin/env python3

"""Train a video classification model."""
import numpy as np
import pprint
import torch

import os, pickle
import oss2 as oss
import torch.nn as nn

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
from utils.meters import TrainMeter, ValMeter

from models.base.builder import build_model, build_agent
from datasets.base.builder import build_loader, shuffle_dataset

from datasets.utils.mixup import Mixup

logger = logging.get_logger(__name__)



def train_epoch(
    train_loader, model, model_ema, agent, optimizer_model, scaler, train_meter, cur_epoch, mixup_fn, cfg
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        model_ema (model): the ema model to update.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (Config): The global config object.
    """
    # Enable train mode.
    if cfg.TRAIN.ONLY_LINEAR and hasattr(cfg.TRAIN, 'DONOT_UPDATE_BN') and cfg.TRAIN.DONOT_UPDATE_BN:
        for name, module in model.named_modules():
            if 'head' not in name and 'backbone' in name and len(name) > 0:
                module.eval()
                module.requires_grad_(False)
            elif len(name) > 0:
                module.train()
                module.requires_grad_(True)
    else:
        model.train()
    agent.train()
    norm_train = False
    num_norms = 0
    # Examine the training status of the batch norm modules.
    du.synchronize()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            num_norms += 1
            if module.training:
                norm_train = True
    logger.info(f"Norm training: {norm_train if num_norms >0 else 'No norm'}")
    # Separately examine the training status of the batch norm 1D modules,
    # as the batch norm 1D is usually used in heads, which needs to be trained
    # despite of the frozen BN in the backbone.
    norm_train = False
    num_norms = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d)):
            num_norms += 1
            if module.training:
                norm_train = True
    logger.info(f"Norm 1d training: {norm_train if num_norms >0 else 'No norm'}")
    train_meter.iter_tic()
    data_size = len(train_loader)

    if hasattr(cfg.AGENT, 'LOSS_FUNC'):
        loss_func = cfg.AGENT.LOSS_FUNC
    else:
        loss_func = 'Loss_MaeAgent'
    random_logits_epoch = cfg.AGENT.RANDOM_LOGITS_EPOCH if hasattr(cfg.AGENT, "RANDOM_LOGITS_EPOCH") else 0
    try:
        agent.network.test_mode = False
    except:
        agent.module.network.test_mode = False
    for cur_iter, (inputs, labels, indexes, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.        
        if misc.get_num_gpus(cfg):
            if not cfg.AUGMENTATION.USE_GPU:
                for k, v in inputs.items():
                    if k == "sentences":
                        continue
                    if isinstance(inputs[k], list):
                        for i in range(len(inputs[k])):
                            if type(v[i]) is torch.Tensor:
                                inputs[k][i] = v[i].cuda(non_blocking=True)
                    elif isinstance(inputs[k], torch.Tensor):
                        inputs[k] = v.cuda(non_blocking=True)
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
                labels["supervised"] = labels["supervised"].cuda()
            if cfg.PRETRAIN.ENABLE:
                for k, v in labels["self-supervised"].items():
                    labels["self-supervised"][k] = v.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        if cfg.AUGMENTATION.BATCH_AUG.ENABLE:
            labels['supervised'] = labels['supervised'].repeat_interleave(inputs['video'].shape[1], dim=0)
            inputs['video'] = inputs['video'].flatten(0, 1)

        # perform mixup on the input
        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])

        # Update the learning rate.
        lr_model = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer_model, lr_model)
        if model_ema is not None:
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                ema_feat = model_ema(inputs)
            agent_output = agent(ema_feat)
        else:
            agent_output = agent(inputs)
        inputs['mask'] = agent_output['mask']
        # logger.info(f"{agent_output['reward_weight'].max().item()}, {agent_output['reward_weight'].min().item()}")
        # Perform the forward pass.
        loss_in_parts = {}
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            model_output = model(inputs)
            loss_model, loss_in_parts_model = losses.Loss_MaeCls(cfg, agent_output, model_output, None, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)

        loss_in_parts.update(loss_in_parts_model)
        # check Nan Loss.
        misc.check_nan_losses(loss_model)

        # Perform the backward pass.
        optimizer_model.zero_grad()
        scaler.scale(loss_model).backward()
        scaler.unscale_(optimizer_model)
        if cfg.OPTIMIZER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.OPTIMIZER.CLIP_GRAD_VAL
            )
        elif cfg.OPTIMIZER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.OPTIMIZER.CLIP_GRAD_L2NORM
            )
        if hasattr(cfg.AGENT, "CLIP_GRAD_L2NORM") and cfg.AGENT.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                agent.parameters(), cfg.AGENT.CLIP_GRAD_L2NORM
            )
        elif hasattr(cfg.AGENT, "CLIP_GRAD_VAL") and cfg.AGENT.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                agent.parameters(), cfg.AGENT.CLIP_GRAD_VAL
            )
        # Update the parameters.
        scaler.step(optimizer_model)
        scaler.update()
        if model_ema is not None:
            if isinstance(model_ema, nn.parallel.DistributedDataParallel):
                model_ema.module.update(model)
            else:
                model_ema.update(model)

        if misc.get_num_gpus(cfg) > 1:
            loss_model = du.all_reduce([loss_model])[0]
        if loss_in_parts is None:
            loss_in_parts = {}
        # loss_in_parts["agent_grad_norm"] = torch.norm(torch.stack([torch.norm(p.grad.detach()) if p.grad is not None else torch.Tensor([0]) for p in agent.parameters()]))
        top1_err, top5_err = None, None
        if isinstance(labels["supervised"], dict):
            raise NotImplemented
        else:
            # Compute the errors.
            preds = model_output['preds_cls'][0]
            num_topks_correct = metrics.topks_correct(preds, labels["supervised"], (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Gather all the predictions across all the devices.
            if misc.get_num_gpus(cfg) > 1:
                loss_model, top1_err, top5_err = du.all_reduce(
                    [loss_model, top1_err, top5_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss_model, top1_err, top5_err = (
                loss_model.item(),
                top1_err.item(),
                top5_err.item(),
            )
        train_meter.iter_toc()
        # Update and log stats.
        train_meter.update_stats(
            top1_err, top5_err, loss_model, lr_model, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
        )
        train_meter.update_custom_stats(loss_in_parts)
            
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, agent, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model/model_ema to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (Config): The global config object.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    agent.eval()
    try:
        agent.network.test_mode = True
    except:
        agent.module.network.test_mode = True
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if misc.get_num_gpus(cfg):
            # Transferthe data to the current GPU device.
            if not cfg.AUGMENTATION.USE_GPU:
                for k, v in inputs.items():
                    if k == "sentences":
                        continue
                    if isinstance(inputs[k], list):
                        for i in range(len(inputs[k])):
                            if isinstance(inputs[k], torch.Tensor):
                                inputs[k][i] = v[i].cuda(non_blocking=True)
                    elif isinstance(inputs[k], torch.Tensor):
                        inputs[k] = v.cuda(non_blocking=True)
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
                labels["supervised"] = labels["supervised"].cuda()
            if cfg.PRETRAIN.ENABLE:
                for k, v in labels["self-supervised"].items():
                    labels["self-supervised"][k] = v.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        if cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'MoSIGenerator'):
            raise NotImplemented
        else:
            agent_output = agent(inputs)
            inputs['mask'] = agent_output['mask']
            model_output = model(inputs)
            preds = model_output['preds_cls'][0]
            top1_err, top5_err = None, None
            if isinstance(labels["supervised"], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels["supervised"], (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if misc.get_num_gpus(cfg) > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                inputs["video"][0].size(0) if cfg.VIDEO.ENABLE else inputs["audio"][0].size(0)
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (Config): The global config object.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)
    agent, agent_ema = build_agent(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        from time import sleep
        sleep(3.0)
        misc.log_agent_model_info(agent, model, cfg, use_train_input=True)

    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    # Construct the optimizer.
    optimizer_model = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer_model, model_bucket, scaler if cfg.TRAIN.MIXED_PRECISION else None)

    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val") if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg) if val_loader is not None else None

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None
    
    if cfg.AUGMENTATION.LABEL_SMOOTHING > 0.0:
        logger.info("Enabling label smoothing.")
        cfg.TRAIN.LOSS_FUNC = "soft_target"

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.OPTIMIZER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    for cur_epoch in range(start_epoch, cfg.OPTIMIZER.MAX_EPOCH, cfg.TRAIN.NUM_FOLDS):

        # Shuffle the dataset.
        shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, model_ema, agent, optimizer_model, scaler, train_meter, cur_epoch, mixup_fn, cfg
        )
        torch.cuda.empty_cache()

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            cu.save_checkpoint_dqn(cfg.OUTPUT_DIR, model, model_ema, agent, optimizer_model, cur_epoch+cfg.TRAIN.NUM_FOLDS-1, cfg, model_bucket, scaler if cfg.TRAIN.MIXED_PRECISION else None)
        if misc.is_eval_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            val_meter.set_model_ema_enabled(False)
            eval_epoch(val_loader, model, agent, val_meter, cur_epoch+cfg.TRAIN.NUM_FOLDS-1, cfg)

    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )
    du.synchronize()

