import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from models import init_biencoder_components
from options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger
from utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    
    cfg.is_DPR_checkpoint=True
    if cfg.is_DPR_checkpoint:
        logger.info("This is a DPR checkpoint")
        saved_state = load_states_from_checkpoint(cfg.model_file)  #['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset', 'epoch', 'encoder_params']
        set_cfg_params_from_state(saved_state.encoder_params, cfg)
        
    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))
    cfg.encoder.pretrained_model_cfg = 'LCRCheck/model/bin/Lawformer'
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)#为什么有三个返回参数
    













if __name__ == "__main__":
    main()