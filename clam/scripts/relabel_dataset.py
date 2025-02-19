"""
Loads CLAM and dataset, computes latent actions 
between each timestep and saves latent actions into the dataset.

For regular CLAM models, we predict the latent action between the last two timesteps.
e.g. context, o_t, o_{t+1} -> la_t

For transformer models, we predict the latent action between each timestep in sequence. 
e.g. o_t-1, o_t, o_{t+1} -> la_t-1, la_t
"""

import json
import uuid
from pathlib import Path

import hydra
import numpy as np
import tensorflow as tf
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from clam.models.clam_utils import get_clam_cls, get_la_dim
from clam.resolvers import *
from clam.trainers import trainer_to_cls
from clam.utils.data_utils import Batch
from clam.utils.dataloader import get_dataloader
from clam.utils.general_utils import to_device, to_numpy
from clam.utils.logger import log


@hydra.main(version_base=None, config_name="relabel_dataset", config_path="../cfg")
def main(cfg: DictConfig):
    # load CLAM from checkpoint
    log("---------------- Loading CLAM from checkpoint ----------------")

    # Load pretrained model using trainer cls
    lam_ckpt = cfg.lam_ckpt
    cfg_file = Path(cfg.lam_ckpt) / "config.yaml"
    log(f"loading config from {cfg_file}", "blue")
    model_cfg = OmegaConf.load(cfg_file)
    model_cfg.load_from_ckpt = True
    model_cfg.ckpt_step = None
    model_cfg.ckpt_file = cfg.lam_ckpt

    # select all of the data
    model_cfg.data.train_frac = 1.0
    model_cfg.data.num_trajs = -1
    model_cfg.data.num_examples = -1
    model_cfg.data.shuffle = False
    task = model_cfg.env.datasets[0]
    model_cfg.env.datasets = cfg.datasets

    # some checks
    transformer_model = (
        "transformer" in model_cfg.name or model_cfg.name == "st_vivit_clam"
    )
    if transformer_model:
        # TODO: fix this, maybe we want some action chunking / temporal ensemble thing
        log(f"Transformer model detected, seq len: {model_cfg.data.seq_len}", "yellow")

        if model_cfg.name == "st_vivit_clam":
            model_cfg.data.shift = model_cfg.data.seq_len - 1

            # we need to shift 1 less than the seq_len
            assert (
                model_cfg.data.seq_len == model_cfg.data.shift + 1
            ), "seq_len must be shift + 1"
            model_cfg.data.pad_dataset = True  # pad the dataset too

        else:
            # this is just a regular transformer
            model_cfg.data.shift = model_cfg.data.seq_len - 1
            model_cfg.data.pad_dataset = True

    trainer = trainer_to_cls[model_cfg.name](model_cfg)
    la_dim = get_la_dim(trainer.cfg)

    LATENT_ACTIONS_DICT = {
        "latent_actions": tf.TensorSpec(shape=(None, la_dim), dtype=np.float32),
    }
    cfg = trainer.cfg

    for ds_name, dataset in trainer.train_ds.items():
        log(f"-------------- Relabeling dataset {ds_name} --------------", "green")
        data_iter = dataset.as_numpy_iterator()

        latent_actions = []
        dones = []
        timesteps = []

        for batch in tqdm.tqdm(data_iter, desc="Relabeling dataset"):
            batch = to_device(batch, trainer.device)
            batch = Batch(**batch)
            
            # make sure to set the model to eval mode
            with torch.no_grad():
                if transformer_model:
                    clam_output = trainer.model(
                        batch.observations,
                        timesteps=batch.timestep,
                        states=batch.states,
                    )
                else:
                    clam_output = trainer.model(batch.observations)

                la = clam_output.la

                if transformer_model:
                    # predict latent actions between each timestep
                    la = la[:, 1:]

            latent_actions.append(to_numpy(la))
            if transformer_model:
                dones.append(to_numpy(batch.is_terminal[:, :-1]))
                timesteps.append(to_numpy(batch.timestep[:, :-1]))
            else:
                dones.append(to_numpy(batch.is_terminal[:, -1]))

        latent_actions = np.concatenate(latent_actions, axis=0)
        dones = np.concatenate(dones, axis=0)
        timesteps = np.concatenate(timesteps, axis=0)

        if transformer_model:
            # flatten latent actions and dones
            latent_actions = latent_actions.reshape(-1, la_dim)
            dones = dones.reshape(-1)
            timesteps = timesteps.reshape(-1)

        # split latent_actions based on dones
        latent_actions = np.split(latent_actions, np.where(dones)[0] + 1)
        timesteps = np.split(timesteps, np.where(dones)[0] + 1)
        dones = np.split(dones, np.where(dones)[0] + 1)
        last_len = latent_actions[-1].shape[0]
        latent_actions = latent_actions[:-1]  # remove last element
        timesteps = timesteps[:-1]

        # little hack for transformer models
<<<<<<< HEAD
        if transformer_model:
            # remove the padded latent actions
            for i in range(1, len(latent_actions)):
                # first = latent_actions[i][0].reshape(1, -1)
                latent_actions[i] = latent_actions[i][last_len:]
                timesteps[i] = timesteps[i][last_len:]
                # latent_actions[i] = np.concatenate([first, latent_actions[i]], axis=0)

            # do it for the first one too
            # latent_actions[0] = np.concatenate([first, latent_actions[0]], axis=0)

        else:
            # apply padding
            for i in range(len(latent_actions)):
                first = latent_actions[i][0].reshape(1, -1)
                last = latent_actions[i][-1].reshape(1, -1)
                latent_actions[i] = np.concatenate(
                    [first, latent_actions[i], last], axis=0
                )
=======
        if model_cfg.env.env_name == "calvin":
            if transformer_model:
                for i in range(1, len(latent_actions)):
                    # find index of 1 in timesteps
                    # remove padding steps in dataset
                    idx = np.where(timesteps[i] == 1)[0][0]
                    latent_actions[i] = latent_actions[i][idx-1:]
                    timesteps[i] = timesteps[i][idx-1:]
        else:
            if transformer_model:
                # remove the padded latent actions
                for i in range(1, len(latent_actions)):
                    # first = latent_actions[i][0].reshape(1, -1)
                    latent_actions[i] = latent_actions[i][last_len:]
                    timesteps[i] = timesteps[i][last_len:]
                    # latent_actions[i] = np.concatenate([first, latent_actions[i]], axis=0)
>>>>>>> soda

                # do it for the first one too
                # latent_actions[0] = np.concatenate([first, latent_actions[0]], axis=0)

            else:
                # apply padding
                for i in range(len(latent_actions)):
                    first = latent_actions[i][0].reshape(1, -1)
                    last = latent_actions[i][-1].reshape(1, -1)
                    latent_actions[i] = np.concatenate(
                        [first, latent_actions[i], last], axis=0
                    )

            # check that all the latent actions are the same length
            la_lengths = [la.shape[0] for la in latent_actions]
            assert len(set(la_lengths)) == 1, "latent actions must be the same length"

        def generator():
            for la in latent_actions:
                yield {"latent_actions": la}

        data_dir = Path(cfg.data.data_dir) / "tensorflow_datasets"

        # create a file for storing the latent action id to file mapping
        mapping_file = data_dir / cfg.env.dataset_name / ds_name / "la_map.json"

        # first read the file and see if it exists
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                la_map = json.load(f)
        else:
            la_map = {}

        if lam_ckpt not in la_map:
            # create a unique id
            id_ = str(uuid.uuid4().fields[-1])[:5]
            la_map[lam_ckpt] = id_
        else:
            id_ = la_map[lam_ckpt]
            log(
                f"la dataset [{id_}] already exists for {lam_ckpt}, overwriting the dataset",
                "yellow",
            )

        with open(mapping_file, "w") as f:
            json.dump(la_map, f)

        save_file = data_dir / cfg.env.dataset_name / ds_name / f"latent_actions_{id_}"

        log(
            f"-------------- Saving latent actions to {save_file} --------------",
            "yellow",
        )
        la_tfds = tf.data.Dataset.from_generator(
            generator, output_signature=LATENT_ACTIONS_DICT
        )
        tf.data.experimental.save(la_tfds, str(save_file))

    # Try loading the saved dataset
    log("-------------- Loading dataset with latent actions --------------", "yellow")
    model_cfg.data.load_latent_actions = True
    model_cfg.data.data_type = "transitions"
    train_ds, eval_ds = get_dataloader(
        model_cfg, model_cfg.env.datasets, model_cfg.env.dataset_split
    )

    batch = next(train_ds[ds_name].as_numpy_iterator())
    for key, value in batch.items():
        log(f"{key}: {value.shape}")


if __name__ == "__main__":
    main()
