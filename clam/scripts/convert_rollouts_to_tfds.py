import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import hydra
import cv2
import einops
import numpy as np
import torch
import tqdm
from r3m import load_r3m
from torchvision import transforms as T

from clam.scripts.data import save_dataset
from clam.utils.logger import log
from clam.scripts.tdmpc2_utils import TASK_SET


class R3MPreprocessor:
    def __init__(self, r3m_id: str):
        self.r3m = load_r3m(r3m_id).to("cuda")
        self.transforms = T.Compose([T.Resize(256), T.CenterCrop(224)])

    def __call__(self, image):
        image = self.transforms(image)
        image = self.r3m(image)
        return image


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../cfg")
def main(cfg):
    fps = [
        # (os.path.join(data_dir, f"{task_name}-30000.pt"), "random"),
        # (os.path.join(data_dir, f"{task_name}-120000.pt"), "medium"),
        (Path(cfg.data_dir) / f"{cfg.task_name}-{cfg.ckpt_num}.pt", "expert"),
    ]
    if cfg.precompute_embeddings:
        log("Precomputing embeddings using R3M")
        preproc = R3MPreprocessor("resnet50")

    for fp, expertise in tqdm.tqdm(fps, desc="Loading data"):
        td = torch.load(fp)

        trajectories = []
        returns = []
        lengths = []

        # skip the first timestep
        observations = td["obs"].numpy()
        actions = td["action"].numpy()
        rewards = td["reward"].numpy()

        # this is N, T, F, H, W, C
        images = td["image"].numpy()

        if cfg.black_white:
            images = images.mean(axis=-1, keepdims=True)

        # we might want to take a different framestack here
        images = images[:, :, -cfg.framestack :]

        # convert these to trajectories
        num_trajs = observations.shape[0]
        log(f"Number of trajectories: {num_trajs} in {fp}")

        for indx in tqdm.tqdm(range(num_trajs), desc="trajectories"):
            traj_imgs = images[indx]
            T = traj_imgs.shape[0]

            if len(images[indx].shape) == 5:
                # flatten first two dimension
                traj_imgs = einops.rearrange(traj_imgs, "t f h w c -> (t f) h w c")

            # convert to tensor
            traj_imgs = torch.tensor(traj_imgs).float().to("cuda")

            # make sure channel first
            traj_imgs = traj_imgs.permute(0, 3, 1, 2)

            # apply the R3M preprocessing here with batches
            # NOTE: only do this if we are not doing BW

            if not cfg.black_white and cfg.precompute_embeddings:
                embeddings = []
                batches = [
                    traj_imgs[i : i + cfg.batch_size]
                    for i in range(0, len(traj_imgs), cfg.batch_size)
                ]
                for batch in batches:
                    with torch.no_grad():
                        embeddings.append(preproc(batch))

                embeddings = torch.cat(embeddings).cpu().numpy()

                # unflatten first two dims
                embeddings = einops.rearrange(embeddings, "(t f) d -> t f d", t=T)
            else:
                embeddings = None

            final_data = {
                "observations": observations[indx],
                "actions": actions[indx][:, :4],  # MW actions are 4D
                "rewards": rewards[indx],
                "images": images[indx],
                "discount": np.ones_like(rewards[indx]),
                "is_last": np.zeros_like(rewards[indx]),
                "is_first": np.zeros_like(rewards[indx]),
                "is_terminal": np.zeros_like(rewards[indx]),
            }

            final_data["is_last"][-1] = 1
            final_data["is_terminal"][-1] = 1
            final_data["is_first"][0] = 1

            if cfg.precompute_embeddings:
                final_data["embeddings"] = embeddings

            trajectories.append(final_data)

            episode_return = np.sum(rewards[indx][1:])
            returns.append(episode_return)
            lengths.append(len(rewards[indx]))

        save_file = (
            Path(cfg.tfds_data_dir) / cfg.dataset_name / f"{cfg.task_name}-{expertise}"
        )

        log(f"Saving to {save_file}")
        save_dataset(
            trajectories,
            save_file,
            env_name="metaworld",
            save_imgs=True,
            black_white=cfg.black_white,
            framestack=cfg.framestack,
            precompute_embeddings=cfg.precompute_embeddings,
        )


if __name__ == "__main__":
    main()
