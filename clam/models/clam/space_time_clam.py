from typing import Tuple

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from clam.models.base import BaseModel
from clam.models.clam.clam import get_vq_cls
from clam.models.clam.transformer_clam import TransformerCLAM

# from clam.models.space_time_attn.models import STTransformer
from clam.models.space_time_attn.models_v2 import STTransformer
from clam.models.space_time_attn.utils import patchify, unpatchify
from clam.models.utils.transformer_utils import get_pos_encoding
from clam.models.utils.utils import CLAMOutput, IDMOutput, compute_perplexity
from clam.utils.logger import log


def extract_state_info(states: torch.Tensor):
    pos_goal = states[:, :, -3:]
    curr_obs_and_prev_obs = states[:, :, :-3]
    curr_obs = curr_obs_and_prev_obs[:, :, : int(curr_obs_and_prev_obs.shape[-1] // 2)]

    hand_pos_gripper = curr_obs[:, :, :4]
    return hand_pos_gripper


class SpaceTimeIDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: Tuple[int, int, int], la_dim: int):
        super().__init__(cfg=cfg, input_dim=input_dim)
        self.name = "SpaceTimeIDM"

        self.la_dim = la_dim
        C, H, W = input_dim
        # the 0th dimension is the channel dimension
        self.patch_token_dim = C * self.cfg.patch_size**2
        self.model_dim = self.cfg.net.dim_model

        # the output_dim should be the latent dimension
        # the sequence dim is [T * N]
        # make sure H and W are divisible by patch_size
        assert H % self.cfg.patch_size == 0 and W % self.cfg.patch_size == 0
        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)

        # also encode the hand position
        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

        self.input_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.encoder = STTransformer(cfg=self.cfg.net)
        self.activation = nn.LeakyReLU(0.2)

        self.action_in = nn.Parameter(torch.randn(1, 1, 1, self.model_dim))

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.la_head = nn.Linear(self.model_dim, self.la_dim)

    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        if self.cfg.quantize_la:
            vq_cls = get_vq_cls(self.cfg.vq.name)
            log(f"Using vq {self.cfg.vq.name}", "green")
            self.cfg.vq.kwargs.dim = self.cfg.la_dim
            self.vq = vq_cls(**self.cfg.vq.kwargs)
        else:
            log("Not using vq, continuous latent action space", "red")

    def forward(
        self, observations, timesteps: torch.Tensor, states: torch.Tensor, **kwargs
    ) -> IDMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            states: [B, T, D] tensor
        """
        B, T, *_ = observations.shape

        # need to put channel last for patchify
        observations = observations.permute(0, 1, 3, 4, 2)

        # [B, T, N, E] where N is the number of patches
        # and E is the patch token dimension
        patches = patchify(observations, self.cfg.patch_size)

        # embed the patches
        patches_embed = self.input_embed(patches)
        patches_embed = self.activation(patches_embed)

        # HMM, adding the action token after i embed the patches
        if self.cfg.add_action_token:
            B, T, N, E = patches_embed.shape

            # add a dummy token to represent the latent actions
            action_pad = self.action_in.expand(B, T, 1, self.model_dim)

            # prepend the action token to the patches
            # [B, T, N+1, E]
            patches_embed = torch.cat([action_pad, patches_embed], dim=2)

        # create temporal embeddings using timesteps
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps.long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N + 1)
        else:
            t_pos_embed = None

        # create spatial embeddings for each patch
        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N + 1).to(patches_embed.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed

        if self.cfg.concatenate_gripper_state:
            # extract the hand position and gripper state
            # embed the hand position and concatenate with the patches
            hand_pos_gripper = extract_state_info(states)
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=N + 1
            )
            patches_embed = self.input_embed_two(
                torch.cat([patches_embed, hand_pos_gripper_embed], dim=-1)
            )

        # [B, T, N+1, E]
        z = self.encoder(patches_embed, pos_embed=pos_embed, causal=False)

        # reshape back to [B, T, N+1, E]
        z = z.view(B, T, -1, self.model_dim)

        if self.cfg.add_action_token:
            # the first element is the action token
            la_z = z[:, :, 0]
        else:
            la_z = z.mean(dim=2)

        la = self.la_head(la_z)

        # TODO: clean this up, duplicated with regular CLAM
        if self.cfg.quantize_la:
            quantized_la, indices, vq_loss = self.vq(la)

            if self.cfg.use_quantized_las:  # replace la with quantized_la
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
                encoder_out=patches,
            )

        # return patches to use in the FDM
        return IDMOutput(la=la, encoder_out=patches)


class SpaceTimeFDM(BaseModel):
    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super().__init__(cfg, input_dim)
        self.name = "SpaceTimeFDM"
        C, H, W = input_dim
        self.patch_token_dim = C * self.cfg.patch_size**2

        self.num_patches = (H // self.cfg.patch_size) * (W // self.cfg.patch_size)
        self.seq_len = (self.cfg.seq_len - 1) * self.num_patches
        log(
            f"Patch token dim: {self.patch_token_dim}, Num patches: {self.num_patches}, Sequence length: {self.seq_len}"
        )
        self.model_dim = self.cfg.net.dim_model

        self.decoder = STTransformer(cfg=self.cfg.net)

        self.patch_embed = nn.Linear(self.patch_token_dim, self.model_dim)
        self.la_embed = nn.Linear(la_dim, self.model_dim)

        self.spatial_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )
        self.temporal_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.cond_pos_embed = get_pos_encoding(
            self.cfg.net.pos_enc, embedding_dim=self.model_dim, max_len=200
        )

        self.to_recon = nn.Linear(self.model_dim, self.patch_token_dim)

        # also encode the hand position
        if self.cfg.concatenate_gripper_state:
            self.hand_pos_embed = nn.Linear(4, self.model_dim)
            self.input_embed_two = nn.Linear(self.model_dim * 2, self.model_dim)

    def forward(
        self,
        observations,
        idm_output: IDMOutput,
        timesteps: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ) -> CLAMOutput:
        """
        Args:
            observations: [B, T, C, H, W] tensor
            idm_output: IDMOutput
        Returns:
            video_recon: [B, T, C, H, W] tensor
        """
        B, T, C, H, W = observations.shape

        # [B, T-1, la_dim]
        la = idm_output.la[:, 1:]

        # [B, T, N, E]
        patches = idm_output.encoder_out

        la_embed = self.la_embed(la)
        # add dimension
        la_embed = einops.rearrange(la_embed, "B T E -> B T 1 E")

        # don't feed in the last timestep, need to be causal
        patches_embed = self.patch_embed(patches[:, :-1])

        video_action_patches = la_embed + patches_embed

        if self.cfg.concatenate_gripper_state:
            # extract the hand position and gripper state
            # embed the hand position and concatenate with the patches
            hand_pos_gripper = extract_state_info(states[:, :-1])
            hand_pos_gripper_embed = self.hand_pos_embed(hand_pos_gripper)
            hand_pos_gripper_embed = einops.repeat(
                hand_pos_gripper_embed, "B T E -> B T N E", N=self.num_patches
            )
            video_action_patches = self.input_embed_two(
                torch.cat([video_action_patches, hand_pos_gripper_embed], dim=-1)
            )

        B, T_minus_one, N, E = video_action_patches.shape

        # create temporal embeddings using timesteps
        if self.cfg.net.pos_enc == "learned":
            t_pos_embed = self.temporal_pos_embed(timesteps[:, :-1].long())
            t_pos_embed = einops.repeat(t_pos_embed, "B T E -> B T N E", N=N)
        else:
            t_pos_embed = None

        # create spatial embeddings
        if self.cfg.net.pos_enc == "learned":
            spatial_coord = torch.arange(N).to(video_action_patches.device)
            spatial_pos_embed = self.spatial_pos_embed(spatial_coord.long())
            spatial_pos_embed = einops.repeat(
                spatial_pos_embed, "N E -> B T N E", B=B, T=T - 1
            )
        else:
            spatial_pos_embed = None

        pos_embed = spatial_pos_embed + t_pos_embed

        if self.cfg.net.pos_enc == "learned":
            cond_pos_embed = self.cond_pos_embed(timesteps[:, 1:].long())
            cond_pos_embed = einops.repeat(cond_pos_embed, "B T E -> B T N E", N=N)
        else:
            cond_pos_embed = None

        # [B, (T-1), N, E]
        video_recon = self.decoder(
            video_action_patches,
            pos_embed=pos_embed,
            causal=True,
            cond=la_embed,  # apply cross attention with the learned latent action
            cond_pos_embed=cond_pos_embed,
        )

        video_recon = self.to_recon(video_recon)

        # reshape back to [B, T-1, N, E]
        video_recon = video_recon.view(B, T - 1, -1, self.patch_token_dim)

        # [B, T-1, H, W, C]
        video_recon = unpatchify(video_recon, self.cfg.patch_size, H, W)

        # put channel first
        video_recon = einops.rearrange(video_recon, "B T H W C -> B T C H W")
        return video_recon


class SpaceTimeCLAM(TransformerCLAM):
    """
    Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VAE model to encode video frames into continuous latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    """

    def __init__(self, cfg: DictConfig, input_dim: int, la_dim: int):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.name = "ST-ViVit"
        self.la_dim = la_dim
        self.idm = SpaceTimeIDM(cfg.idm, input_dim=input_dim, la_dim=la_dim)
        self.fdm = SpaceTimeFDM(cfg.fdm, input_dim=input_dim, la_dim=la_dim)

    def forward(
        self,
        observations,
        timesteps: torch.Tensor,
        states: torch.Tensor = None,
        **kwargs,
    ) -> CLAMOutput:
        return super().forward(observations, timesteps=timesteps, states=states)
