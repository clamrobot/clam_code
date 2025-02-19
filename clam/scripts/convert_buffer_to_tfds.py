"""
python3 scripts/convert_buffer_to_tfds.py \
    task_name=mw-assembly \
    ckpt_num=100000 \
    data_dir=/scr/shared/clam/datasets/metaworld/tdmpc2_buffer_imgs \
    tfds_data_dir=/scr/shared/clam/tensorflow_datasets \
    dataset_name=tdmpc2-buffer-imgs \
    save_imgs=True \
    framestack=3
"""

import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm

from clam.scripts.data import save_dataset
from clam.scripts.tdmpc2_utils import TASK_SET
from clam.utils.logger import log


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../cfg")
def main(cfg):
    replay_buffer = Path(cfg.data_dir) / f"buffer_{cfg.task_name}_{cfg.ckpt_num}.pt"
    td = torch.load(str(replay_buffer))

    trajectories = []
    returns = []
    lengths = []

    # skip the first timestep
    observations = td["obs"].cpu().numpy()
    actions = td["action"].cpu().numpy()
    rewards = td["reward"].cpu().numpy()
    episodes = td["episode"].cpu().numpy()
    # task = td["task"].cpu().numpy()

    save_imgs = cfg.save_imgs and "image" in td

    if cfg.save_imgs and "image" not in td:
        raise ValueError("Cannot save images when they are not in the buffer")

    # for buffer data, this is N, C, H, W
    if save_imgs:
        images = td["image"].cpu().numpy()

    # split the data into episodes, episode is a list of ep_idx, e.g. [0, 0, 1, 1, ...]
    episode_split = np.where(episodes[1:] != episodes[:-1])[0] + 1

    # split the data into episodes
    observations = np.split(observations, episode_split)
    actions = np.split(actions, episode_split)
    rewards = np.split(rewards, episode_split)
    episodes = np.split(episodes, episode_split)

    # remove episodes that are not the same length
    # TODO; figure out why some episodes are not the same length
    episode_lens = [len(ep) for ep in observations]
    valid_episodes_id = np.where(np.array(episode_lens) == episode_lens[100])[0]

    observations = [observations[i] for i in valid_episodes_id]
    actions = [actions[i] for i in valid_episodes_id]
    rewards = [rewards[i] for i in valid_episodes_id]
    episodes = [episodes[i] for i in valid_episodes_id]

    if save_imgs:
        images = np.split(images, episode_split)
        images = [images[i] for i in valid_episodes_id]

    if cfg.framestack > 1 and save_imgs:
        log(f"Framestacking images with framestack {cfg.framestack}")
        for ep in range(len(images)):
            ep_imgs = images[ep]

            # first pad with the first frame framestack - 1 times
            first_frame = ep_imgs[0]
            first_frame = np.expand_dims(first_frame, axis=0)
            first_frame = np.repeat(first_frame, cfg.framestack - 1, axis=0)
            ep_imgs = np.concatenate([first_frame, ep_imgs], axis=0)

            # then create the framestack
            new_imgs = []
            for i in range(len(ep_imgs) - cfg.framestack + 1):
                new_imgs.append(ep_imgs[i : i + cfg.framestack])

            images[ep] = np.array(new_imgs)

    # convert these to trajectories
    num_trajs = len(valid_episodes_id)

    for indx in tqdm.tqdm(range(num_trajs), desc="trajectories"):
        # skip the first timestep
        obs = observations[indx][:-1]
        act = actions[indx][1:, :4]
        rew = rewards[indx][1:]

        if save_imgs:
            image = images[indx][:-1]

        final_data = {
            "observations": obs,
            "actions": act,  # MW actions are 4D
            "rewards": rew,
            "discount": np.ones_like(rew),
            "is_last": np.zeros_like(rew),
            "is_first": np.zeros_like(rew),
            "is_terminal": np.zeros_like(rew),
        }

        final_data["is_last"][-1] = 1
        final_data["is_terminal"][-1] = 1
        final_data["is_first"][0] = 1

        if save_imgs:
            final_data["images"] = image

        trajectories.append(final_data)

        episode_return = np.sum(rew)
        returns.append(episode_return)
        lengths.append(len(rew))

    # save full set of trajectories
    save_file = Path(cfg.tfds_data_dir) / cfg.dataset_name / cfg.task_name
    save_dataset(
        trajectories,
        save_file,
        env_name=cfg.env_name,
        save_imgs=save_imgs,
        framestack=cfg.framestack,
    )

    # we want to keep the random / medium data which is anything less than max_return
    trajs = [traj for traj, ret in zip(trajectories, returns) if ret < cfg.max_return]
    sorted_returns = np.argsort(returns)
    # trajs = [trajectories[i] for i in sorted_returns[-num_samples:]]

    log(f"Keeping {len(trajs)} trajectories with return < {cfg.max_return}")
    save_file = Path(cfg.tfds_data_dir) / cfg.dataset_name / f"{cfg.task_name}-al"
    save_dataset(
        trajs,
        save_file,
        env_name="metaworld",
        save_imgs=save_imgs,
        framestack=cfg.framestack,
    )

    # ======= EXPERT TRAJECTORIES =======
    expert_trajs = [trajectories[i] for i in sorted_returns[-cfg.num_samples :]]
    save_file = Path(cfg.tfds_data_dir) / cfg.dataset_name / f"{cfg.task_name}-expert"
    save_dataset(
        expert_trajs,
        save_file,
        env_name="metaworld",
        save_imgs=save_imgs,
        framestack=cfg.framestack,
    )

    # ======= RANDOM MED TRAJECTORIES =======
    random_med = [trajectories[i] for i in sorted_returns[: cfg.num_samples]]
    save_file = (
        Path(cfg.tfds_data_dir)
        / cfg.dataset_name
        / f"{cfg.task_name}-rm-{cfg.num_samples}"
    )
    save_dataset(
        random_med,
        save_file,
        env_name="metaworld",
        save_imgs=save_imgs,
        framestack=cfg.framestack,
    )

    random_med = [trajectories[i] for i in sorted_returns[: cfg.num_samples * 2]]
    save_file = (
        Path(cfg.tfds_data_dir)
        / cfg.dataset_name
        / f"{cfg.task_name}-rm-{cfg.num_samples * 2}"
    )
    save_dataset(
        random_med,
        save_file,
        env_name="metaworld",
        save_imgs=save_imgs,
        framestack=cfg.framestack,
    )


if __name__ == "__main__":
    main()
