import collections
import glob
import logging
import os
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",
        "optimizer_dict",
        "scheduler_dict",
        "offset",
        "epoch",
        "encoder_params",
    ],
)

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)

print('开始了')
a=load_states_from_checkpoint('LCRCheck/model/sourceCode/LEAD/LEAD_checkpoints/dpr_biencoder_70.pth')
print(a.encoder_params)
print(a.model_dict.keys())