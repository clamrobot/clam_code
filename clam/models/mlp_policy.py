from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from clam.models.base import BaseModel
from clam.models.utils.utils import make_conv_net, make_impala_cnn, make_mlp
from clam.utils.logger import log


class MLPPolicy(BaseModel):
    def __init__(
        self,
        cfg: DictConfig,
        input_dim: Union[int, tuple],
        output_dim: int = None,
        feature_extractor: nn.Module = None,
    ):
        super(MLPPolicy, self).__init__(cfg, input_dim)
        self.name = "MLPPolicy"

        if feature_extractor is not None:
            # we use a pre-trained feature extractor
            self.input_embedding = feature_extractor
            if hasattr(feature_extractor, "output_dim"):
                embedding_output_dim = feature_extractor.output_dim
            elif isinstance(feature_extractor, nn.Sequential):
                embedding_output_dim = feature_extractor[-1].out_features
        else:
            # encode the input
            if cfg.image_obs and not cfg.use_pretrained_embeddings:
                if cfg.encoder.name == "impala_cnn":
                    self.input_embedding, embedding_output_dim = make_impala_cnn(
                        input_dim,
                        output_embedding_dim=cfg.embedding_dim,
                        net_kwargs=cfg.encoder,
                    )
                else:
                    self.input_embedding, embedding_output_dim = make_conv_net(
                        input_dim,
                        output_embedding_dim=cfg.embedding_dim,
                        net_kwargs=cfg.encoder,
                        apply_output_head=True,
                    )
            else:
                self.input_embedding = nn.Linear(input_dim, cfg.embedding_dim)
                embedding_output_dim = cfg.embedding_dim

        # policy network

        self.gaussian_output = False
        if hasattr(cfg, "gaussian_policy") and cfg.gaussian_policy:
            self.gaussian_output = True
            output_dim *= 2

        self.policy, _ = make_mlp(
            input_dim=embedding_output_dim,
            net_kwargs=cfg.net,
            output_dim=output_dim,
        )

        if hasattr(cfg, "action_activation") and cfg.action_activation:
            self.action_activation = getattr(nn, cfg.action_activation)()
            log(f"Using action activation: {cfg.action_activation}", "red")
        else:
            self.action_activation = None

    def forward(self, x):
        input_embed = self.input_embedding(x)

        if self.gaussian_output:
            model_output = self.policy(input_embed)
            mean, logvar = torch.chunk(model_output, 2, dim=-1)

            # apply action scale
            if self.action_activation:
                mean = self.action_activation(mean)

                if self.cfg.action_scale:
                    mean = mean * self.cfg.action_scale

            # clamp logvar
            logvar = torch.clamp(logvar, min=-20, max=2)
            return mean, logvar

        policy_out = self.policy(input_embed)
        if self.action_activation:
            policy_out = self.action_activation(policy_out)

            # scale action
            if self.cfg.action_scale:
                policy_out = policy_out * self.cfg.action_scale

        return policy_out
