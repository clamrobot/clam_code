import torch
import torch.nn.functional as F
import einops


def patchify(videos, size: int):
    B, T, H, W, C = videos.shape
    H, W = videos.shape[2:4]
    padding_height = -H % size
    padding_width = -W % size

    # Pad with zeros (default behavior of F.pad)
    x = F.pad(videos, (0, 0, 0, padding_width, 0, padding_height, 0, 0, 0, 0))
    return einops.rearrange(
        x, "b t (hn hp) (wn wp) c -> b t (hn wn) (hp wp c)", hp=size, wp=size
    )


def unpatchify(patches, size: int, h_out: int, w_out: int):
    h_pad = -h_out % size
    hn = (h_out + h_pad) // size
    x = einops.rearrange(
        patches,
        "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
        hp=size,
        wp=size,
        hn=hn,
    )
    return x[:, :, :h_out, :w_out]
