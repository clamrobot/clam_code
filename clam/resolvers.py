from functools import partial
from typing import List

import numpy as np
from omegaconf import OmegaConf


def resolve_clam_name(
    lam_hp_name, joint_training, joint_extra, apply_vq, vq_hp_name
) -> str:
    if joint_training:
        lam_hp_name += f"_{joint_extra}"
    if apply_vq:
        lam_hp_name += f"_vk-{vq_hp_name}"
    return lam_hp_name


def filter_keys(key_val, keys_to_keep):
    parts = key_val.split("-")
    if len(parts) > 1 and parts[0] in keys_to_keep:
        return True
    return False


def resolve_ckpt_name(
    base_hp_name: str, clam_ckpt_path: str, ad_ckpt_path: str = None
) -> str:
    alg = clam_ckpt_path.split("/")[4]
    clam_base = clam_ckpt_path.split("/")[-1]

    # filter for specific keys
    keys = ["AL", "la", "nt", "vq", "adw"]
    clam_base = list(
        filter(partial(filter_keys, keys_to_keep=keys), clam_base.split("_"))
    )
    # join them back
    clam_base = alg + "_" + "_".join(clam_base)

    if ad_ckpt_path:
        ad_base = ad_ckpt_path.split("/")[-1]
        return f"{base_hp_name}_ckpt-{clam_base}_ad-{ad_base}"
    return f"{base_hp_name}_ckpt-{clam_base}"


def resolve_lapa_ckpt_name(
    base_hp_name: str, lapa_ckpt_path: str = None, policy_ckpt_path: str = None
) -> str:
    if lapa_ckpt_path:  # this is for LAP
        clam_base = lapa_ckpt_path.split("/")[-1]

        # filter for specific keys
        keys = ["AL", "la", "nt", "vq"]
        clam_base = list(
            filter(partial(filter_keys, keys_to_keep=keys), clam_base.split("_"))
        )
        clam_base = "_".join(clam_base)
        return f"stage1_{base_hp_name}_clam-{clam_base}"

    if policy_ckpt_path:  # this is for LAPA
        policy_base = policy_ckpt_path.split("/")[-1]
        keys = ["AL", "la", "nt", "vq"]
        policy_base = list(
            filter(partial(filter_keys, keys_to_keep=keys), policy_base.split("_"))
        )
        policy_base = "_".join(policy_base)
        return f"stage2_{base_hp_name}_lap-{policy_base}"  # forget the other thing since it is making the name too long
    return base_hp_name


def resolve_vpt_ckpt_name(base_hp_name: str, vpt_ckpt_path: str) -> str:
    alg = vpt_ckpt_path.split("/")[4]
    vpt_base = vpt_ckpt_path.split("/")[-1]

    # filter for specific keys
    keys = ["nt"]
    vpt_base = list(
        filter(partial(filter_keys, keys_to_keep=keys), vpt_base.split("_"))
    )
    # join them back
    vpt_base = alg + "_" + "_".join(vpt_base)
    return f"{base_hp_name}_ckpt-{vpt_base}"


def fix_ds_name(ds_name: List[str], dataset_name: str, dataset_split: List[int]) -> str:
    ds_names = []

    mapping = {
        "peg-insert-side": "peg-side",
        "assembly": "assem",
        "hammer": "ham",
        "door-open": "do-op",
        "window-open": "win-op",
        "random": "R",
        "medium": "M",
        "expert": "E",
        "mw-": "",
        "relative": "R",
        "absolute": "A",
        "close_drawer": "close-d",
    }

    # shorten the dataset names
    for name in ds_name:
        shortened_name = name.split("/")[-1]

        # replace the name with the mapping
        for key, val in mapping.items():
            if key in shortened_name:
                shortened_name = shortened_name.replace(key, val)

        ds_names.append(shortened_name)

    ds_name = "-".join(ds_names)
    ds_name = dataset_name + "-" + ds_name
    ds_name = ds_name + "-".join([str(x) for x in dataset_split])

    return ds_name


def fix_env_hp_name(
    hp_name: str = "", image_obs: bool = False, image_extra: str = ""
) -> str:
    if image_obs:
        if hp_name == "":
            hp_name = image_extra
        else:
            hp_name += f"-{image_extra}"

    return hp_name


OmegaConf.register_new_resolver("multiply", lambda a, b: a * b)
OmegaConf.register_new_resolver("concat", lambda l: ",".join(l[:2]))
OmegaConf.register_new_resolver("resolve_clam_name", resolve_clam_name)
OmegaConf.register_new_resolver("resolve_ckpt_name", resolve_ckpt_name)
OmegaConf.register_new_resolver("resolve_lapa_ckpt_name", resolve_lapa_ckpt_name)
OmegaConf.register_new_resolver("eval", lambda x: eval(x))
OmegaConf.register_new_resolver("fix_ds_name", fix_ds_name)
OmegaConf.register_new_resolver("fix_env_hp_name", fix_env_hp_name)
OmegaConf.register_new_resolver("resolve_vpt_ckpt_name", resolve_vpt_ckpt_name)
