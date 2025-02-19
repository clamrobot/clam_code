"""
Scripts convert d4rl dataset to tfds format.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from clam.envs.utils import make_envs
from clam.scripts.data import save_dataset
from clam.utils.logger import log


def main():
    data_dir = Path("/scr/shared/clam")
    ds_name = "d4rl_mujoco_halfcheetah/v2-expert"
    env_id = "halfcheetah-expert-v2"

    ds = tfds.load(ds_name, split="train", data_dir=data_dir / "datasets")
    trajectories = []

    def episode_to_steps(episode):
        # setting the size to max length of all the trajectories
        episode = rlds.transformations.batch(
            episode["steps"], 2000, shift=0, drop_remainder=False
        )
        return episode

    num_trajs = len(ds)
    ds = ds.flat_map(episode_to_steps)
    ds = ds.as_numpy_iterator()

    returns = []
    lengths = []

    render_dataset = True

    if render_dataset:
        dummy_env = make_envs(env_name="mujoco", num_envs=1, env_id=env_id)

    # iterate through dataset
    for traj_idx, example in enumerate(
        tqdm.tqdm(ds, total=num_trajs, desc="trajectories")
    ):
        if traj_idx > 300:
            break

        observations = example["observation"]
        actions = example["action"]
        rewards = example["reward"]

        final_data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "discount": np.ones_like(rewards),
            "is_last": np.zeros_like(rewards),
            "is_first": np.zeros_like(rewards),
            "is_terminal": np.zeros_like(rewards),
            "env_state": np.concatenate(
                [example["infos"]["qpos"], example["infos"]["qvel"]], axis=-1
            ),
        }

        final_data["is_last"][-1] = 1
        final_data["is_terminal"][-1] = 1
        final_data["is_first"][0] = 1

        if render_dataset:
            dummy_env.reset()
            imgs = []

            for ts in tqdm.tqdm(range(len(observations)), desc="rendering images"):
                qpos = example["infos"]["qpos"][ts]
                qvel = example["infos"]["qvel"][ts]

                dummy_env.envs[0].set_state(qpos, qvel)
                img = dummy_env.envs[0].render(mode="rgb_array")
                # resize image to 64x64
                img = cv2.resize(img, (64, 64))
                imgs.append(img)

            final_data["images"] = np.stack(imgs)

        episode_return = np.sum(rewards)
        returns.append(episode_return)
        lengths.append(len(rewards))

        trajectories.append(final_data)

    log(f"Number of trajectories: {len(trajectories)}")
    log(f"Average return: {np.mean(returns)}")
    log(f"Average length: {np.mean(lengths)}")

    if render_dataset:
        ds_name += "_images"

    save_file = Path(data_dir) / "tensorflow_datasets" / ds_name

    save_dataset(
        trajectories, save_file, env_name="halfcheetah", save_imgs=render_dataset
    )


if __name__ == "__main__":
    main()
