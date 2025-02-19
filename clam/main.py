import os
import sys

# hide warning messages
import warnings

import torch

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# remove tensorflow debug messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# prevent jax from preallocating all gpu memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# make sure that the model training is deterministic
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["WANDB_DIR"] = "/scr/matthewh6/wandb"

import tensorflow as tf

from clam.utils.logger import log

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.config.experimental.set_visible_devices([], "GPU")

import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from resolvers import *

from clam.trainers import trainer_to_cls
from clam.utils.general_utils import omegaconf_to_dict, print_dict


@hydra.main(version_base=None, config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    overrides = HydraConfig.get().overrides.task

    log(f"Job overrides: {overrides}")

    # remove tensorflow debug messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # prevent jax from preallocating all gpu memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # make sure that the model training is deterministic
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.config.experimental.set_visible_devices([], "GPU")

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg.name not in trainer_to_cls:
        raise ValueError(f"Invalid trainer name: {cfg.name}")

    log("start")
    trainer = trainer_to_cls[cfg.name](cfg)
    trainer.train()

    # end program
    log("end")
    sys.exit(0)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    main()
