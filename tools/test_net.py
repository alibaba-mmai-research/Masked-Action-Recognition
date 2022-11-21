#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import json

import utils.bucket as bu
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.misc as misc
from datasets.base.builder import build_loader
from models.base.builder import build_model, build_agent
from utils.meters import TestMeter, EpicKitchenMeter

logger = logging.get_logger(__name__)

def save_features(model_output, video_name, labels, root):
    encoder_logits_backbone = model_output['encoder_logits_backbone'].mean(dim=1).detach().cpu()
    deocoder_logits = model_output['preds_cls'][1].detach().cpu()
    pred_logits = model_output['preds_cls'][0].softmax(dim=-1).detach().cpu()
    if os.path.exists(root) is False:
        os.makedirs(root)
    for b in range(encoder_logits_backbone.size(0)):
        name = video_name[b]
        label = labels['supervised'][b]
        torch.save([encoder_logits_backbone[b], deocoder_logits[b], pred_logits[b], label], f"{root}/{name}.pkl")


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, agent=None):
    """
    Perform multi-view test on the specified test set, where {cfg.TEST.NUM_ENSEMBLE_VIEWS}
    clips and {cfg.TEST.NUM_SPATIAL_CROPS} crops are sampled temporally and spatially, forming 
    in total cfg.TEST.NUM_ENSEMBLE_VIEWS x cfg.TEST.NUM_SPATIAL_CROPS views.
    The softmax scores are aggregated by summation. 
    The predictions are compared with the ground-truth labels and the accuracy is logged.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (Config): The global config object.
    """
    # Enable eval mode.
    model.eval()
    if agent is not None:
        agent.eval()
        try:
            agent.network.test_mode = True
        except:
            agent.module.network.test_mode = True
    test_meter.iter_tic()
    res_dic = {}
    for cur_iter, (inputs, labels, video_idx, meta) in enumerate(test_loader):
        if misc.get_num_gpus(cfg):
            # Transfer the data to the current GPU device.
            for k, v in inputs.items():
                if not isinstance(v, (torch.Tensor, list)):
                    continue
                if isinstance(inputs[k], list):
                    for i in range(len(inputs[k])):
                        if isinstance(inputs[k], torch.Tensor):
                            inputs[k][i] = v[i].cuda(non_blocking=True)
                elif isinstance(inputs[k], torch.Tensor):
                    inputs[k] = v.cuda(non_blocking=True)

            # Transfer the labels to the current GPU device.
            if isinstance(labels["supervised"], dict):
                for k, v in labels["supervised"].items():
                    labels["supervised"][k] = v.cuda()
            else:
                labels["supervised"] = labels["supervised"].cuda()
            video_idx = video_idx.cuda()
            if cfg.PRETRAIN.ENABLE:
                for k, v in labels["self-supervised"].items():
                    labels["self-supervised"][k] = v.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        if cfg.PRETRAIN.ENABLE:
            raise NotImplemented
        else:
            # Perform the forward pass.
            if agent is not None:
                agent_output = agent(inputs)
                inputs['mask'] = agent_output['mask']
            model_output = model(inputs)
            # save_features(model_output, inputs['name'], labels, root='/home/admin/workspace/project/code/transformer-long-video/DAMO-Action/output/ssv2-feat/ft_vit_base_v29-l')
            preds = model_output['preds_cls'][0]
            if len(preds.shape) == 3:
                preds = preds.mean(dim=1)
            preds = preds.softmax(dim=-1)
            if cfg.DATA.MULTI_LABEL:
                # Mainly for the EPIC-KITCHENS dataset.
                if misc.get_num_gpus(cfg) > 1:
                    preds_verb, preds_noun, labels_verb, labels_noun, video_idx = du.all_gather(
                        [
                            preds["verb_class"], 
                            preds["noun_class"], 
                            labels["supervised"]["verb_class"], 
                            labels["supervised"]["noun_class"], 
                            video_idx
                        ]
                    )
                else:
                    preds_verb  = preds["verb_class"]
                    preds_noun  = preds["noun_class"]
                    labels_verb = labels["supervised"]["verb_class"]
                    labels_noun = labels["supervised"]["noun_class"]
                if misc.get_num_gpus(cfg):
                    preds_verb  = preds_verb.cpu()
                    preds_noun  = preds_noun.cpu()
                    labels_verb = labels_verb.cpu()
                    labels_noun = labels_noun.cpu()
                    video_idx   = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds_verb.detach(),
                    preds_noun.detach(),
                    labels_verb.detach(),
                    labels_noun.detach(),
                    video_idx.detach(),
                    [test_loader.dataset._get_sample_info(i)["name"] for i in video_idx.tolist()] if "name" in test_loader.dataset._get_sample_info(0).keys() else []
                )
                test_meter.log_iter_stats(cur_iter)
            else:

                # Gather all the predictions across all the devices to perform ensemble.
                if misc.get_num_gpus(cfg) > 1:
                    preds, labels_supervised, video_idx = du.all_gather(
                        [preds, labels["supervised"], video_idx]
                    )
                else:
                    labels_supervised = labels["supervised"]
                if misc.get_num_gpus(cfg):
                    preds = preds.cpu()
                    labels_supervised = labels_supervised.cpu()
                    video_idx = video_idx.cpu()

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(
                    preds.detach(), labels_supervised.detach(), video_idx.detach()
                )
                test_meter.log_iter_stats(cur_iter)
        test_meter.iter_tic()

    # save epic-kitchens statistics
    if "epickitchen100" in cfg.TEST.DATASET:
        if cfg.DATA.MULTI_LABEL or not hasattr(cfg.DATA, "TRAIN_VERSION"):
            verb = test_meter.video_preds["verb_class"]
            noun = test_meter.video_preds["noun_class"]
            file_name_verb = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            file_name_noun = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(verb, file_name_verb)
            torch.save(noun, file_name_noun)

            logger.info(
                "Successfully saved verb and noun results to {} and {}.".format(file_name_verb, file_name_noun)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_verb":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_verb" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved verb results to {}.".format(file_name)
            )
        elif hasattr(cfg.DATA, "TRAIN_VERSION") and cfg.DATA.TRAIN_VERSION == "only_train_noun":
            file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE.split('.')[0]+"_noun" + f"{'_ema' if test_meter.model_ema_enabled else ''}" + ".pyth")
            torch.save(test_meter.video_preds, file_name)
            logger.info(
                "Successfully saved noun results to {}.".format(file_name)
            )
    elif cfg.TEST.SAVE_PREDS: 
        file_name = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_PREDS_PATH.split('.')[0] + "_{}clipsx{}crops.pyth".format(
            cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
        ))
        torch.save(test_meter.video_preds, file_name)
        logger.info(f"Successfully saved preds to {file_name}.")
    if agent is not None:
        logger.info(f"Performance with agent:")
    test_meter.finalize_metrics()
    test_meter.reset()


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (Config): The global config object.
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Test with config:")
        logger.info(cfg)

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg)
    agent, agent_ema = build_agent(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
    model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)

    cu.load_test_checkpoint(cfg, model, model_ema, model_bucket)

    # Create video testing loaders.
    test_loader = build_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    assert (
        len(test_loader.dataset)
        % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
        == 0
    )
    # Create meters for multi-view testing.
    cfg.LOG_PERIOD = max(len(test_loader) // 10, 5)
    if cfg.DATA.MULTI_LABEL or hasattr(cfg.DATA, "TRAIN_VERSION"):
        test_meter = EpicKitchenMeter(
            cfg,
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )
    else:
        test_meter = TestMeter(
            cfg,
            len(test_loader.dataset)
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.VIDEO.HEAD.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Perform multi-view test on the entire dataset.
    test_meter.set_model_ema_enabled(False)
    perform_test(test_loader, model, test_meter, cfg, agent)
    # perform_test(test_loader, model, test_meter, cfg)
    if model_ema is not None:
        test_meter.set_model_ema_enabled(True)
        # perform_test(test_loader, model_ema.module, test_meter, cfg)
        perform_test(test_loader, model_ema.module, test_meter, cfg, agent)
    return
    # upload results to bucket
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

        result_file_name = cfg.TEST.LOG_FILE
        result_file_name = result_file_name.split('.')[0] + "_res_ema" + ".json"
        filename = os.path.join(cfg.OUTPUT_DIR, result_file_name)
        if os.path.exists(filename):
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )

        if cfg.TEST.SAVE_PREDS: 
            filename = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_PREDS_PATH.split('.')[0] + "_{}clipsx{}crops.pyth".format(
                cfg.TEST.NUM_ENSEMBLE_VIEWS, cfg.TEST.NUM_SPATIAL_CROPS
            ))
            bu.put_to_bucket(
                model_bucket, 
                cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
                filename,
                cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
            )

    # synchronize all processes on different GPUs to prevent collapsing
    du.synchronize()