import dataclasses
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torch import nn

from clam.utils.logger import log


@dataclasses.dataclass
class VQOutput:
    loss: torch.Tensor
    indices: torch.Tensor
    quantized: torch.Tensor
    la: torch.Tensor


@dataclasses.dataclass
class IDMOutput:
    la: torch.Tensor
    quantized_la: torch.Tensor = None
    vq_loss: torch.Tensor = None
    vq_metrics: Dict = None
    vq_outputs: Dict = None
    encoder_out: torch.Tensor = None


@dataclasses.dataclass
class CLAMOutput:
    la: torch.Tensor
    reconstructed_obs: torch.Tensor
    idm_output: IDMOutput


@dataclasses.dataclass
class HierarchicalIDMOutput(IDMOutput):
    discrete_la: torch.Tensor = None
    continuous_la: torch.Tensor = None


@dataclasses.dataclass
class DiffusionCLAMOutput:
    noise: torch.Tensor
    reconstructed_obs: torch.Tensor
    la: torch.Tensor


def compute_perplexity(indices, codebook_size: int):
    indices_count = torch.bincount(indices.view(-1), minlength=codebook_size)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.all_reduce(indices_count)
    avg_probs = indices_count.float() / indices_count.sum()

    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp().item()
    return perplexity


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def make_mlp(
    net_kwargs: DictConfig,
    input_dim: int,
    output_dim: int = None,
    activation: nn.Module = nn.LeakyReLU(0.2),
    use_batchnorm: bool = False,
):
    mlp = []

    prev_dim = input_dim
    for hidden_dim in net_kwargs.hidden_dims:
        mlp.append(nn.Linear(prev_dim, hidden_dim))

        if use_batchnorm:  # Add BatchNorm if the flag is set
            mlp.append(nn.BatchNorm1d(hidden_dim))

        mlp.append(activation)
        prev_dim = hidden_dim

    if output_dim is not None:
        mlp.append(nn.Linear(prev_dim, output_dim))

    output_dim = prev_dim if output_dim is None else output_dim
    mlp = nn.Sequential(*mlp)
    return mlp, output_dim


class ResidualLayer(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_out_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return x + self.res_block(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        conv_kwargs: Dict,
        output_channel: int,
        batch_norm: bool = False,
        residual_layer: bool = False,
        activation: nn.Module = nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.conv = nn.Conv2d(**conv_kwargs)
        self.batch_norm = (
            nn.BatchNorm2d(output_channel) if batch_norm else nn.Identity()
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = activation

        if residual_layer:
            self.residual_layer = ResidualLayer(
                in_out_dim=output_channel, hidden_dim=output_channel // 2
            )
        else:
            self.residual_layer = None

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        # apply residual here
        if self.residual_layer is not None:
            x = self.residual_layer(x)
        x = self.max_pool(x)
        x = self.activation(x)
        return x


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim: List[int],
        net_kwargs: Dict,
        activation: nn.Module = nn.LeakyReLU(0.2),
        apply_output_head: bool = False,
        output_embedding_dim: int = None,
    ):
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        out_chs = net_kwargs.out_channels
        kernel_sizes = net_kwargs.kernel_size
        strides = net_kwargs.stride
        paddings = net_kwargs.padding
        in_ch = input_dim[0]
        self.apply_output_head = apply_output_head

        for out_ch, kernel_size, stride, padding in zip(
            out_chs, kernel_sizes, strides, paddings
        ):
            self.conv_blocks.append(
                ConvBlock(
                    conv_kwargs=dict(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    output_channel=out_ch,
                    batch_norm=net_kwargs.batch_norm,
                    residual_layer=net_kwargs.residual_layer,
                    activation=activation,
                )
            )
            in_ch = out_ch

        # compute the output size of the CNN
        log("Computing the output size of the CNN", "yellow")
        log("=" * 50, "yellow")
        with torch.no_grad():
            x = torch.zeros(1, *input_dim)
            log(f"input size: {x.size()}")
            for i, conv_block in enumerate(self.conv_blocks):
                x = conv_block(x)
                log(f"layer {i + 1}, output size: {x.size()}")
        log("=" * 50, "yellow")

        if apply_output_head:
            conv_out_size = x.flatten().size(-1)
            self.fc = nn.Linear(conv_out_size, output_embedding_dim)
            self.output_size = output_embedding_dim
            log(f"apply output head, output dim: {conv_out_size}", "yellow")
        else:
            self.output_size = conv_out_size = x.size()[1:]
            log(f"returning feature maps, output dim: {self.output_size}", "yellow")

    def forward(self, x, return_intermediates: bool = False):
        # also return the intermediate outputs for the U-Net training
        intermediates = []

        for conv_block in self.conv_blocks:
            x = conv_block(x)
            intermediates.append(x)

        if self.apply_output_head:
            x = x.flatten(start_dim=1)
            x = self.fc(x)

        if return_intermediates:
            return x, intermediates

        return x


def make_conv_net(
    input_dim: Tuple,
    net_kwargs: DictConfig,
    output_embedding_dim: int,
    apply_output_head: bool = False,
    activation: nn.Module = nn.LeakyReLU(0.2),
):
    log(
        f"Making conv net, input dim: {input_dim}, output dim: {output_embedding_dim}",
        "yellow",
    )
    # channel must be first
    conv_net = ConvNet(
        input_dim=input_dim,
        net_kwargs=net_kwargs,
        activation=activation,
        apply_output_head=apply_output_head,
        output_embedding_dim=output_embedding_dim,
    )
    output_size = conv_net.output_size
    return conv_net, output_size


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        conv_kwargs: Dict,
        batch_norm: bool = False,
        residual_layer: bool = False,
        activation: nn.Module = nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(**conv_kwargs)
        self.batch_norm = (
            nn.BatchNorm2d(conv_kwargs["out_channels"]) if batch_norm else nn.Identity()
        )
        if residual_layer:
            self.residual_layer = ResidualLayer(
                in_out_dim=conv_kwargs["out_channels"],
                hidden_dim=conv_kwargs["out_channels"] // 2,
            )
        else:
            self.residual_layer = None

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.residual_layer is not None:
            x = self.residual_layer(x)
        x = self.activation(x)
        return x


class UpConvNet(nn.Module):
    def __init__(
        self,
        input_dim: List[int],
        net_kwargs: Dict,
        output_channels: int,
        action_dim: int = None,
        state_dim: int = None,
        activation: nn.Module = nn.LeakyReLU(0.2),
    ):
        super().__init__()

        self.upconv_blocks = nn.ModuleList()
        out_chs = net_kwargs.out_channels
        kernel_sizes = net_kwargs.kernel_size
        strides = net_kwargs.stride
        paddings = net_kwargs.padding
        in_ch = input_dim[0]

        for i, (out_ch, kernel_size, stride, padding) in enumerate(
            zip(out_chs, kernel_sizes, strides, paddings)
        ):
            self.upconv_blocks.append(
                ConvTransposeBlock(
                    conv_kwargs=dict(
                        in_channels=in_ch + out_ch
                        if i != 0
                        else in_ch + action_dim,  # TODO:  fix this
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    batch_norm=net_kwargs.batch_norm,
                    residual_layer=net_kwargs.residual_layer,
                    activation=activation,
                )
            )
            in_ch = out_ch

        # compute the output size of the CNN
        log("Computing the output size of the upconv CNN", "yellow")
        log("=" * 50, "yellow")
        with torch.no_grad():
            x = torch.zeros(1, *input_dim)
            log(f"input size: {x.size()}")
            for i, upconv_block in enumerate(self.upconv_blocks):
                # add intermediate outputs from encoder
                if i == 0:
                    intermediate = torch.zeros(1, action_dim, *x.size()[2:])
                else:
                    intermediate = torch.zeros(1, out_chs[i], *x.size()[2:])
                x = torch.cat([x, intermediate], dim=1)
                x = upconv_block(x)
                log(f"layer {i + 1}, output size: {x.size()}")
        log("=" * 50, "yellow")

        self.output_size = x.size()[1:]
        final_channel = x.size()[1]
        log(f"returning feature maps, output dim: {self.output_size}", "yellow")

        # one final layer to get the correct output channel dimension
        self.final_conv = nn.Conv2d(
            final_channel + state_dim, output_channels, kernel_size=3, padding=1
        )

    def forward(
        self, x, intermediates: List[torch.Tensor] = None, state: torch.Tensor = None
    ):
        for i, upconv_block in enumerate(self.upconv_blocks):
            # concatenate with intermediate outputs from the encoder
            if intermediates:
                # make sure the intermediate output has the same spatial size
                assert x.size()[2:] == intermediates[i].size()[2:]
                x = torch.cat([x, intermediates[i]], dim=1)

            x = upconv_block(x)

        x = self.final_conv(torch.cat([x, state], dim=1))
        return x


def make_upconv_net(
    input_dim: Tuple,
    output_channels: int,
    net_kwargs: DictConfig,
    action_dim: int = None,
    state_dim: int = None,
    activation: nn.Module = nn.LeakyReLU(0.2),
):
    log(
        f"Making upconv net, input dim: {input_dim}, output channels: {output_channels}",
        "yellow",
    )
    upconv_net = UpConvNet(
        input_dim=input_dim,
        net_kwargs=net_kwargs,
        output_channels=output_channels,
        action_dim=action_dim,
        state_dim=state_dim,
        activation=activation,
    )
    return upconv_net


class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ImpalaResidualBlock(output_channel)
        self.res_block1 = ImpalaResidualBlock(output_channel)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x


def make_impala_cnn(
    input_dim: Tuple,
    output_embedding_dim: int,
    net_kwargs: DictConfig,
    activation: nn.Module = nn.LeakyReLU(0.2),
):
    # channel must be first
    input_channel = input_dim[0]

    conv_stack = []

    for out_ch in net_kwargs.hidden_dim:
        conv_seq = ConvSequence(
            input_channel=input_channel, output_channel=net_kwargs.cnn_scale * out_ch
        )
        input_channel = net_kwargs.cnn_scale * out_ch
        conv_stack.append(conv_seq)

    conv_stack.append(nn.Flatten())

    # compute the output size of the CNN
    with torch.no_grad():
        x = torch.zeros(1, *input_dim)
        for layer in conv_stack:
            x = layer(x)
        conv_out_size = x.size(-1)

    fc = nn.Linear(in_features=conv_out_size, out_features=output_embedding_dim)
    conv_stack = nn.Sequential(*conv_stack, nn.ReLU(), fc)
    return conv_stack, output_embedding_dim
