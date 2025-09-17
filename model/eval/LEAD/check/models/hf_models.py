import logging
from typing import Tuple, List

import torch
import transformers
from torch import Tensor as T
from torch import nn


if transformers.__version__.startswith("4"):
    from transformers import AutoModel, AutoConfig, AutoTokenizer, LongformerModel, BertModel
    transformers.logging.set_verbosity_error()
    from transformers import AdamW
    from transformers import BertTokenizer
    from transformers import RobertaTokenizer


logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    logger.info(f"you are currently using code for longformer, if this is not your intention, please refer to please refer to the /LEAD/dpr/models/hf_models.py")
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,            #lawformer
        projection_dim=cfg.encoder.projection_dim,   # 0
        dropout=dropout,                             # 0.1
        pretrained=cfg.encoder.pretrained,           # True
        **kwargs
    )  #看起来就是两个lawformer
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )
    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False  #False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer

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