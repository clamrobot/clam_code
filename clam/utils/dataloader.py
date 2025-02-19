import json
import os
from functools import partial
from pathlib import Path
from typing import List

import einops
import rlds
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from clam.utils.logger import log

os.environ["TFDS_DATA_DIR"] = "/scr/shared/prompt_dtla/tensorflow_datasets"


def episode_to_step_custom(episode, size, shift):
    episode = tf.data.Dataset.from_tensor_slices(episode)
    return rlds.transformations.batch(
        episode, size=size, shift=shift, drop_remainder=True
    )


# add additional fields to the dataset
def add_new_fields(x):
    x["mask"] = tf.ones_like(x["actions"])
    x["timestep"] = tf.range(tf.shape(x["actions"])[0])
    return x


def use_image_observations(
    x,
    channel_first: bool = False,
    use_pretrained_embeddings: bool = False,
    image_shape: List[int] = [84, 84],
):
    if "embeddings" in x and use_pretrained_embeddings:
        x["observations"] = x["embeddings"]

        has_framestack = len(x["observations"].shape) == 3

        if has_framestack:
            # flatten the dimensions
            x["observations"] = einops.rearrange(x["observations"], "B F D -> B (F D)")

    elif "images" in x:
        state = x["observations"]
        x["observations"] = x["images"]
        x["states"] = state  # add the state information here in case we want to use it

        if channel_first:
            # has framestack
            has_framestack = len(x["observations"].shape) == 5

            if has_framestack:
                # take the last frame first
                x["observations"] = einops.rearrange(
                    x["observations"], "B F H W C -> B (F C) H W"
                )
            else:
                # reshape images here
                # TODO: remove this later
                x["observations"] = tf.image.resize(
                    x["observations"], OmegaConf.to_container(image_shape)
                )
                x["observations"] = tf.transpose(x["observations"], perm=[0, 3, 1, 2])

        if tf.reduce_max(x["observations"]) > 1:
            x["observations"] = tf.cast(x["observations"], tf.float32) / 255.0
        else:
            x["observations"] = tf.cast(x["observations"], tf.float32)
    return x


def process_state(x, cfg, env_name):
    states = x["observations"]
    has_framestack = len(states.shape) == 3

    if has_framestack:
        # take the last state!
        states = states[:, -1]

    x["observations"] = states
    x["states"] = states

    # for calvin, add scene_obs
    if env_name == "calvin":
        x["observations"] = tf.concat([x["observations"], x["scene_obs"]], axis=-1)

    return x


def pad_dataset(x, pad):
    # create a dataset from tensor with padded shapes
    for key in x:
        x[key] = tf.concat([x[key], pad[key]], axis=0)
    return x


# remove trajectories where the number of steps is less than 2
def filter_fn(traj):
    return tf.math.greater(tf.shape(traj["observations"])[0], 2)


def process_dataset(
    cfg: DictConfig,
    ds: tf.data.Dataset,
    shuffle: bool = True,
    env_name: str = None,
    drop_remainder: bool = False,
    use_pretrained_embeddings: bool = False,
):
    """
    Applies transformations to base tfds such as batching, shuffling, etc.
    """
    ds = ds.filter(filter_fn)

    # caching the dataset makes it faster in the next iteration
    ds = ds.cache()

    # the buffer size is important for memory usage
    # and affects the speed
    # shuffle here is for trajectories
    if shuffle:
        ds = ds.shuffle(100, reshuffle_each_iteration=False)

    # limit the number of trajectories that we use
    ds = ds.take(cfg.num_trajs)
    log(f"\ttaking {cfg.num_trajs} trajectories")

    # compute return of the trajectories
    if "rewards" in ds.element_spec:
        returns = list(
            ds.map(
                lambda episode: tf.reduce_sum(episode["rewards"])
            ).as_numpy_iterator()
        )
        traj_lens = list(
            ds.map(lambda episode: tf.shape(episode["rewards"])[0]).as_numpy_iterator()
        )
        if len(returns) > 0:
            log(
                f"\tN: {len(returns)} | Average return: {sum(returns) / len(returns)} | Max return: {max(returns)} | Min return: {min(returns)} | Average traj len: {sum(traj_lens) / len(traj_lens)}",
                "yellow",
            )

    ds = ds.map(add_new_fields)

    # replace observations with images
    if cfg.image_obs:
        log("replace observations with images", "yellow")
        # we want to convert from [H, W, C] to [C, H, W]
        ds = ds.map(
            partial(
                use_image_observations,
                channel_first=True,
                use_pretrained_embeddings=use_pretrained_embeddings,
                image_shape=cfg.image_shape,
            )
        )
    else:
        ds = ds.map(partial(process_state, cfg=cfg, env_name=env_name))

    # maybe pad the dataset here
    if cfg.pad_dataset:
        # TODO: what happens if we use transitions here
        # add extra timesteps to each trajectory
        log("padding dataset", "yellow")
        pad = rlds.transformations.zeros_from_spec(ds.element_spec)

        def repeat_padding(pad, batch_size):
            return tf.nest.map_structure(
                lambda x: tf.repeat(x, repeats=batch_size, axis=0), pad
            )

        # padding needed when we use shift
        # pad = repeat_padding(pad, cfg.seq_len - 1)
        pad = repeat_padding(pad, 5)
        ds = ds.map(partial(pad_dataset, pad=pad))

    # quick assert

    if cfg.data_type == "n_step":
        assert not cfg.load_latent_actions, (
            "Cannot use n_step with loading latent actions"
        )

        ds = ds.flat_map(
            partial(episode_to_step_custom, size=cfg.seq_len, shift=cfg.shift)
        )
    elif cfg.data_type == "transitions":
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    else:
        raise ValueError(f"unknown data type: {cfg.data_type}")

    # shuffle the full dataset one more time
    if shuffle:  # shuffle here is for transitions
        log("\tshuffling dataset")
        ds = ds.shuffle(1000, reshuffle_each_iteration=False)

    if cfg.num_examples != -1:
        log(f"\ttaking {cfg.num_examples} examples")
        ds = ds.take(cfg.num_examples)

        # recommended to do dataset.take(k).cache().repeat()
        ds = ds.cache()

    ds = ds.batch(cfg.batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataloader(
    cfg: DictConfig,
    dataset_names: List[str],
    dataset_split: List[int],
    shuffle: bool = True,
):
    """
    Returns a dictionary containing the training and validation datasets.
    Validation dataset is a dictionary of {env_id: dataset}
    """
    data_cfg = cfg.data
    data_dir = Path(data_cfg.data_dir) / "tensorflow_datasets"
    log(f"loading tfds dataset from: {data_dir}")

    env_id = cfg.env.env_id

    datasets = {}
    dataset_split = dataset_split[: len(dataset_names)]
    # convert this into a ratio
    dataset_ratio = [x / sum(dataset_split) for x in dataset_split]

    log(
        f"loading dataset for {env_id}, num datasets: {len(dataset_names)}, ratios: {dataset_ratio}"
    )

    total_trajs = 0
    ds_to_len = {}
    for ds_name in dataset_names:
        save_file = data_dir / cfg.env.dataset_name / ds_name
        ds = tf.data.experimental.load(str(save_file))
        log(f"\tdataset name: {ds_name}, num trajs: {len(ds)}")

        if data_cfg.load_latent_actions:
            mapping_file = data_dir / cfg.env.dataset_name / ds_name / "la_map.json"

            if mapping_file.exists():
                log(f"Loading latent actions mapping from {mapping_file}", "yellow")
                with open(mapping_file, "r") as f:
                    la_map = json.load(f)
            else:
                raise ValueError(
                    f"Latent actions mapping file not found: {mapping_file}"
                )

            if not hasattr(cfg, "lam_ckpt"):
                raise ValueError("lam_ckpt not found in config")

            lam_ckpt = cfg.lam_ckpt
            id_ = la_map[lam_ckpt]
            la_file = save_file / f"latent_actions_{id_}"

            log(f"Loading latent actions relabelled from {la_file}", "yellow")

            if la_file.exists():
                latent_actions_ds = tf.data.experimental.load(str(la_file))
            else:
                raise ValueError(f"Latent actions file not found: {la_file}")

            combined_ds = tf.data.Dataset.zip((ds, latent_actions_ds))
            combined_ds = combined_ds.map(
                lambda x, y: {**x, "latent_actions": y["latent_actions"]}
            )
            ds = combined_ds

        datasets[ds_name] = ds
        total_trajs += len(ds)
        ds_to_len[ds_name] = len(ds)

    log(f"total trajectories: {total_trajs}")
    for ds_name in dataset_names:
        log(f"\t{ds_name}: {ds_to_len[ds_name]} trajs")

    # split dataset into train and eval
    train_ds = {}
    eval_ds = {}

    log("split dataset into train and eval: ")
    for i, ds_name in enumerate(dataset_names):
        num_take = int(ds_to_len[ds_name] * cfg.data.train_frac)
        num_eval = ds_to_len[ds_name] - num_take
        log(f"\t{ds_name}: num train trajs: {num_take}, num eval trajs: {num_eval}")
        train_ds[ds_name] = datasets[ds_name].take(num_take)
        eval_ds[ds_name] = datasets[ds_name].skip(num_take)

    log("creating train datasets")
    for i, ds_name in enumerate(dataset_names):
        cfg_train = cfg.data.copy()
        if cfg.data.num_trajs != -1:
            cfg_train.num_trajs = int(cfg.data.num_trajs * dataset_ratio[i])
        if cfg.data.num_examples != -1:
            cfg_train.num_examples = int(cfg.data.num_examples * dataset_ratio[i])

        log(
            f"\t{ds_name}: num_trajs: {cfg_train.num_trajs}, num_examples: {cfg_train.num_examples}"
        )
        train_ds[ds_name] = process_dataset(
            cfg_train,
            train_ds[ds_name],
            env_name=cfg.env.env_name,
            shuffle=shuffle,
            use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
        )

    log("creating eval datasets")
    # use all the trajectories in the eval dataset
    cfg_eval = cfg.data.copy()
    cfg_eval.num_trajs = -1
    cfg_eval.num_examples = -1
    for i, ds_name in enumerate(dataset_names):
        eval_ds[ds_name] = process_dataset(
            cfg_eval,
            eval_ds[ds_name],
            env_name=cfg.env.env_name,
            shuffle=False,
            use_pretrained_embeddings=cfg.model.use_pretrained_embeddings,
        )

    return train_ds, eval_ds


if __name__ == "__main__":
    pass
