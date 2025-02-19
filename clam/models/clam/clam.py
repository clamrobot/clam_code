import math
from typing import List, Tuple, Union

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig
from vector_quantize_pytorch import FSQ, ResidualFSQ, ResidualVQ, VectorQuantize

from clam.models.base import BaseModel
from clam.models.utils.utils import (
    CLAMOutput,
    IDMOutput,
    compute_perplexity,
    make_conv_net,
    make_mlp,
    make_upconv_net,
)
from clam.utils.logger import log


def get_vq_cls(cls_name):
    if cls_name == "residual_fsq":
        return ResidualFSQ
    elif cls_name == "fsq":
        return FSQ
    elif cls_name == "residual":
        return ResidualVQ
    elif cls_name == "ema":
        return VectorQuantize
    else:
        raise ValueError(f"vq_cls: {cls_name} not supported")


class IDM(BaseModel):
    """
    IDM takes (o_t, o_t+1) and returns z_t
    """

    def __init__(self, cfg: DictConfig, input_dim: Union[int, Tuple]):
        super().__init__(cfg, input_dim=input_dim)

        log("---------------------- Initializing IDM ----------------------", "blue")
        if self.cfg.image_obs:
            # either use a trained encoder or load pretrained one
            input_dim = list(input_dim)
            orig_input_dim = input_dim[0]
            input_dim[0] *= self.cfg.context_len + 1

            log(
                f"input computation: {orig_input_dim} * ({self.cfg.context_len} + 1) = {input_dim[0]}"
            )
            self.input_encoder, output_dim = make_conv_net(
                input_dim,
                net_kwargs=self.cfg.net,
                output_embedding_dim=self.cfg.embedding_dim,
                apply_output_head=True,
            )
        else:
            prev_dim = input_dim * (self.cfg.context_len + 1)
            log(
                f"input computation: {input_dim} * ({self.cfg.context_len} + 1) = {prev_dim}"
            )
            self.input_encoder, output_dim = make_mlp(
                net_kwargs=self.cfg.net, input_dim=prev_dim
            )

        if hasattr(self.cfg, "quantize_la") and self.cfg.quantize_la:
            vq_cls = get_vq_cls(self.cfg.vq.name)
            log(f"Using vq {self.cfg.vq.name}", "green")
            self.vq = vq_cls(**self.cfg.vq.kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")

        if hasattr(self.cfg, "distributional_la") and self.cfg.distributional_la:
            self.la_mean = nn.Linear(output_dim, cfg.la_dim)
            self.la_logvar = nn.Linear(output_dim, cfg.la_dim)
        else:
            self.la = nn.Linear(output_dim, cfg.la_dim)

    def forward(self, observations):
        """
        Args:
            observations (..., o_t, o_t+1) [(B * context_len), input_dim]

        Returns:
            la: z_t [B, la_dim]
        """

        if self.cfg.image_obs:
            context = observations[:, : self.cfg.context_len]
            next_observation = observations[:, self.cfg.context_len :]
        else:
            context = observations[:, : self.cfg.context_len].flatten(start_dim=1)

            # TODO: handle when this is more than size of 1
            next_observation = observations[:, self.cfg.context_len :].squeeze(dim=1)

        if self.cfg.image_obs:
            # concatenate along the seq len dimension
            x = torch.cat([context, next_observation], dim=1)

            # flatten the seq len and channel dims
            x = einops.rearrange(x, "b t c h w -> b (t c) h w")
        else:
            x = torch.cat([context, next_observation], dim=-1)

        x = self.input_encoder(x)

        if hasattr(self.cfg, "distributional_la") and self.cfg.distributional_la:
            la_mean = self.la_mean(x)
            la_logvar = self.la_logvar(x)
            # clamp
            la_logvar = torch.clamp(la_logvar, -10, 10)
            la = torch.concat([la_mean, la_logvar], dim=-1)
        else:
            la = self.la(x)

        if hasattr(self.cfg, "quantize_la") and self.cfg.quantize_la:
            quantized_la, indices, vq_loss = self.vq(la)

            if self.cfg.use_quantized_las:
                la = quantized_la

            vq_loss = vq_loss.mean()

            vq_outputs = {"indices": indices}
            vq_metrics = {
                "vq_loss": vq_loss.item(),
                "perplexity": compute_perplexity(
                    indices, self.cfg.vq.kwargs.codebook_size
                ),
            }

            return IDMOutput(
                la=la,
                quantized_la=quantized_la,
                vq_loss=vq_loss,
                vq_metrics=vq_metrics,
                vq_outputs=vq_outputs,
            )

        return IDMOutput(la=la)


class FDM(nn.Module):
    def __init__(self, cfg: DictConfig, input_dim: Union[int, Tuple], la_dim: int):
        super().__init__()
        self.cfg = cfg

        # TODO: should the FDM have its own encoder or
        # share the same encoder as the IDM?

        log("---------------------- Initializing FDM ----------------------", "blue")
        if self.cfg.image_obs:
            # either use a trained decoder or load pretrained one
            input_dim = list(input_dim)
            orig_input_dim = input_dim[0]
            input_dim[0] *= self.cfg.context_len
            input_dim[0] += (
                cfg.la_dim
            )  # we inject the latent action into the input encoder

            log(
                f"input computation: {orig_input_dim} * {self.cfg.context_len} + {cfg.la_dim} = {input_dim[0]}"
            )

            if self.cfg.predict_target_embedding:
                apply_output_head = True
            else:
                apply_output_head = False

            self.input_encoder, output_dim = make_conv_net(
                input_dim,
                net_kwargs=cfg.input_encoder,
                output_embedding_dim=cfg.embedding_dim,
                apply_output_head=apply_output_head,
            )

            if self.cfg.predict_target_embedding:
                # instead of pixel reconstruction, we predict the target image embedding
                self.to_observation, _ = make_mlp(
                    net_kwargs=cfg.net, input_dim=output_dim, output_dim=output_dim
                )
            else:
                # concatenate state for final conv
                # this is a upsampling layer to reconstruct the image
                self.to_observation = make_upconv_net(
                    input_dim=(256, 1, 1),  # final feature map
                    output_channels=3 * self.cfg.n_frame_stack,
                    net_kwargs=cfg.net,
                    action_dim=cfg.la_dim,
                    state_dim=orig_input_dim
                    * self.cfg.context_len,  # this gets added to the final feature map
                )
        else:
            # context + latent action
            prev_dim = input_dim * cfg.context_len + la_dim
            log(
                f"input computation: ({input_dim} * {cfg.context_len}) + {la_dim} = {prev_dim}"
            )
            self.input_encoder, output_dim = make_mlp(
                net_kwargs=cfg.net, input_dim=prev_dim
            )
            self.to_observation = nn.Linear(output_dim, input_dim)

    def forward(
        self, observations, idm_output, intermediates: List[torch.Tensor] = None
    ):
        la = idm_output.la

        if self.cfg.image_obs:
            context = observations[:, : self.cfg.context_len]
        else:
            context = observations[:, : self.cfg.context_len].flatten(start_dim=1)

        if self.cfg.image_obs:
            context = einops.rearrange(context, "b t c h w -> b (t c) h w")
            _, _, h, w = context.shape

            # add the latent action to the context
            la_repeated = einops.repeat(la, "b d -> b d h w", h=h, w=w)
            x = torch.cat([context, la_repeated], dim=1)
            feature_maps, intermediates = self.input_encoder(
                x, return_intermediates=True
            )

            # reverse the intermediates
            intermediates = intermediates[::-1]

            # TODO: replace the first intermediate with action
            la = einops.repeat(la, "b d -> b d h w", h=1, w=1)
            intermediates[0] = la

            # apply upconv layers to reconstruct original image
            reconstructions = self.to_observation(
                feature_maps, intermediates=intermediates, state=context
            )

        else:
            x = torch.cat([context, la], dim=-1)
            encoded_features = self.input_encoder(x)
            reconstructions = self.to_observation(encoded_features)

        return reconstructions


class ContinuousLAM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super().__init__(cfg, input_dim)
        self.name = "CLAM"

        if self.cfg.share_input_encoder:
            import ipdb

            ipdb.set_trace()

        self.la_dim = la_dim
        self.idm = IDM(cfg.idm, input_dim=input_dim)
        self.fdm = FDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

    def forward(self, observations, **kwargs):
        idm_output = self.idm(observations)
        la = idm_output.la

        # first reparameterize the latent action if distributional
        if self.cfg.distributional_la:
            la_mean = la[:, : self.la_dim]
            la_logvar = la[:, self.la_dim :]

            # clamp logvar
            la_logvar = torch.clamp(la_logvar, -10, 10)

            la_sample = la_mean + torch.exp(0.5 * la_logvar) * torch.randn_like(la_mean)
        else:
            la_sample = la

        idm_output.la = la_sample

        reconstructed_obs = self.fdm(observations, idm_output)

        if self.cfg.image_obs and self.cfg.activate_output:
            # apply sigmoid to the output
            # images are normalized between 0 and 1
            # reconstructed_obs = torch.sigmoid(reconstructed_obs)
            reconstructed_obs = (torch.tanh(reconstructed_obs) + 1) / 2

        return CLAMOutput(
            la=la, reconstructed_obs=reconstructed_obs, idm_output=idm_output
        )

    def reparameterize(self, la):
        latent_mean = la[:, : self.cfg.la_dim]
        latent_logvar = la[:, self.cfg.la_dim :]
        la = latent_mean + torch.exp(0.5 * latent_logvar) * torch.randn_like(
            latent_mean
        )
        return la
