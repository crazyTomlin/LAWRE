import collections
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import transformers
from torch import Tensor as T
from torch import nn
from transformers import AutoModel
from transformers import BertTokenizer
from transformers import AutoModel, AutoConfig, AutoTokenizer, LongformerModel, BertModel
from torch.serialization import default_restore_location


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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

pretrained_model='LCRCheck/model/bin/Lawformer'
model_file='LCRCheck/model/sourceCode/LEAD/LEAD_checkpoints/dpr_biencoder.70'

def get_bert_tensorizer(text):
    sequence_length = 2048        
    tokenizer = get_bert_tokenizer()     
    token_ids = tokenizer.encode(
                    text,
                    max_length=2048,
                    pad_to_max_length=False,
                    truncation=True,
                )
    encoded = tokenizer.batch_encode_plus(
        text,  
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )
    
    return torch.tensor(token_ids),encoded

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained('LCRCheck/model/bin/chinese-roberta-wwm-ext', do_lower_case=True)

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info("Reading saved model from %s", model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys())
    return CheckpointState(**state_dict)

class HFBertEncoder(LongformerModel):
    def __init__(self, config, project_dim: int = 0):
        LongformerModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> AutoModel:
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)  #cfg_name=lawformer
        cfg = AutoConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")  #lawformer
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:
        # BertModel.forward()
        global_attention_mask = torch.zeros(
            input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, representation_token_pos] = 1
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )

        # HF >4.0 version support
        # print(type(out))
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.models.longformer.modeling_longformer.LongformerBaseModelOutputWithPooling,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

def main():
    text='我是啦啦啦'
    # token,encode=get_bert_tensorizer(text)
    # print(token)
    # print(encode)
    saved_state = load_states_from_checkpoint(model_file)
    
    question_encoder = HFBertEncoder.init_encoder(
        pretrained_model,            
        projection_dim=0,   
        dropout=0.1,                             
        pretrained=True,           
        #inference_only=True
    )  #看起来就是两个lawformer
    ctx_encoder = HFBertEncoder.init_encoder(
        pretrained_model,            
        projection_dim=0,   
        dropout=0.1,                            
        pretrained=True,           
        #inference_only=True
    )
    
    encoder=question_encoder
    encoder.eval()
    
    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    encoder.load_state_dict(ctx_state, strict=False)
    
  

  
    
    
if __name__ == "__main__":
    main()
