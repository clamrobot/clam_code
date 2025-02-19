import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
import torch
import wandb
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary

import clam.utils.general_utils as gutl
from clam.envs.utils import make_envs
from clam.utils.dataloader import get_dataloader
from clam.utils.general_utils import omegaconf_to_dict
from clam.utils.logger import log


class BaseTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        if cfg.debug:
            log("RUNNING IN DEBUG MODE", "red")
            # set some default config values
            cfg.num_updates = 10
            cfg.num_evals = 1
            cfg.num_eval_steps = 10
            cfg.num_eval_rollouts = 1

        # check if hydraconfig is set

        try:
            hydra_cfg = HydraConfig.get()
        except ValueError:
            hydra_cfg = None

        if hydra_cfg is not None:
            # determine if we are sweeping
            launcher = hydra_cfg.runtime["choices"]["hydra/launcher"]
            sweep = launcher in ["slurm"]
            log(f"launcher: {launcher}, sweep: {sweep}")

        # if we are loading from checkpoint, we don't need to make new dirs
        if self.cfg.load_from_ckpt:
            self.exp_dir = Path(self.cfg.exp_dir)
        else:
            if hydra_cfg and sweep:
                self.exp_dir = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir
            else:
                if not self.cfg.exp_dir:
                    self.exp_dir = Path(hydra_cfg.run.dir)
                else:
                    self.exp_dir = Path(self.cfg.exp_dir) / self.cfg.hp_name

        log(f"experiment dir: {self.exp_dir}")

        # add exp_dir to config
        self.cfg.exp_dir = str(self.exp_dir)

        # set random seeds
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        tf.random.set_seed(cfg.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"using device: {self.device}")

        if self.cfg.mode == "train":
            if not self.cfg.load_from_ckpt:
                self.log_dir = self.exp_dir / "logs"
                self.ckpt_dir = self.exp_dir / "model_ckpts"
                self.video_dir = self.exp_dir / "videos"

                # create directories
                self.ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.video_dir.mkdir(parents=True, exist_ok=True)
                self.log_dir.mkdir(parents=True, exist_ok=True)

                wandb_name = self.cfg.wandb_name

                if self.cfg.use_wandb:
                    self.wandb_run = wandb.init(
                        # set the wandb project where this run will be logged
                        entity=cfg.wandb_entity,
                        project=cfg.wandb_project,
                        name=wandb_name,
                        notes=self.cfg.wandb_notes,
                        tags=self.cfg.wandb_tags,
                        # track hyperparameters and run metadata
                        config=omegaconf_to_dict(self.cfg),
                        group=self.cfg.group_name,
                    )
                    wandb_url = self.wandb_run.get_url()
                    self.cfg.wandb_url = wandb_url  # add wandb url to config
                    log(f"wandb url: {wandb_url}")

                else:
                    self.wandb_run = None

                # save config to yaml file
                OmegaConf.save(self.cfg, f=self.exp_dir / "config.yaml")
        else:
            self.wandb_run = None

        # create env
        # log(f"creating {self.cfg.env.env_name} environments...")

        # self.envs = make_envs(**self.cfg.env)

        if cfg.best_metric == "max":
            self.best_metric = float("-inf")
        else:
            self.best_metric = float("inf")

        log("loading train and eval datasets", "blue")
        # load datasets
        self.train_ds, self.eval_ds = get_dataloader(
            cfg,
            dataset_names=cfg.env.datasets,
            dataset_split=cfg.env.dataset_split,
            shuffle=cfg.data.shuffle,
        )

        # combine them and uniformly sample from them
        self.train_dataloader = tf.data.Dataset.sample_from_datasets(
            list(self.train_ds.values())
        )
        self.eval_dataloader = tf.data.Dataset.sample_from_datasets(
            list(self.eval_ds.values())
        )

        # print batch item shapes
        # determine obs_shape based on the dataset
        batch = next(self.train_dataloader.as_numpy_iterator())

        log("=" * 50)
        log("Shapes of batch items:")
        for k, v in batch.items():
            log(f"{k}: {v.shape}, {v.dtype}, {v.min()}, {v.max()}, {v.mean()}")

        observations = batch["observations"]

        if cfg.env.image_obs and not cfg.model.use_pretrained_embeddings:
            if cfg.data.data_type == "n_step":
                self.obs_shape = observations.shape[2:]
            else:
                self.obs_shape = observations.shape[1:]
        else:
            self.obs_shape = observations.shape[-1]

        log(f"observation shape: {self.obs_shape}")

        # figure out how many update steps between each validation step
        if self.cfg.eval_every != -1:
            self.eval_every = self.cfg.eval_every
        elif self.cfg.num_evals != -1:
            self.eval_every = int(self.cfg.num_updates // self.cfg.num_evals)
        elif self.cfg.eval_perc != -1:
            self.eval_every = int(self.cfg.num_updates * self.cfg.eval_perc)
        else:
            raise ValueError("no eval interval specified")

        log(f"evaluating model every: {self.eval_every}")

        # initialize model
        self.model = self.setup_model()
        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model)

        # initialize optimizer
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler()

        # count number of parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        log("=" * 50)
        log(f"number of parameters: {num_params}")
        log(f"model: {self.model}")

        # Initialize Accelerator
        if self.cfg.accelerate.use:
            log("Initializing Accelerator", "yellow")
            self.accelerator = Accelerator(
                mixed_precision="fp16" if self.cfg.accelerate.use_fp16 else "no"
            )

            # Prepare model, optimizer, dataloaders
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = (
                self.accelerator.prepare(
                    self.model,
                    self.optimizer,
                    self.train_dataloader,
                    self.eval_dataloader,
                )
            )

            if hasattr(self, "action_decoder"):
                (
                    self.action_decoder,
                    self.action_decoder_optimizer,
                ) = self.accelerator.prepare(
                    self.action_decoder, self.action_decoder_optimizer
                )
        else:
            # for mixed precision training
            self.scaler = GradScaler()

        # print model summary
        # if self.cfg.name == "clam" or self.cfg.name == "action_decoder":
        #     pass
        # elif isinstance(self.obs_shape, int):
        #     summary(self.model, (self.obs_shape,))
        # else:
        #     summary(self.model, self.obs_shape)

        # count trainable parameters
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        log(f"number of trainable parameters: {num_trainable_params}")

        # count frozen/untrainable parameters
        num_frozen_params = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        log(f"number of frozen parameters: {num_frozen_params}")

    def setup_logging(self):
        pass

    def setup_model(self):
        pass

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        optimizer = opt_cls(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)

        log(
            f"using opt: {self.cfg.optimizer.name}, scheduler: {self.cfg.lr_scheduler.name}",
            "yellow",
        )

        # make this a sequential LR scheduler with warmstarts
        warmstart_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=self.cfg.optimizer.num_warmup_steps,
        )

        scheduler = scheduler_cls(optimizer, **self.cfg.lr_scheduler.params)

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmstart_scheduler, scheduler],
            milestones=[self.cfg.optimizer.num_warmup_steps],
        )
        return optimizer, scheduler

    def eval(self, step: int):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self, ckpt_dict: Dict, metrics: Dict, iter: int = None):
        # use orbax?
        if self.cfg.save_key and self.cfg.save_key in metrics:
            key = self.cfg.save_key
            if (self.cfg.best_metric == "max" and metrics[key] > self.best_metric) or (
                self.cfg.best_metric == "min" and metrics[key] < self.best_metric
            ):
                self.best_metric = metrics[key]
                ckpt_file = self.ckpt_dir / "best.pkl"
                log(
                    f"new best value: {metrics[key]}, saving best model at epoch {iter} to {ckpt_file}"
                )
                with open(ckpt_file, "wb") as f:
                    pickle.dump(ckpt_dict, f)

                # create a file with the best metric in the name, use a placeholder
                best_ckpt_file = self.ckpt_dir / "best.txt"
                with open(best_ckpt_file, "w") as f:
                    f.write(f"{iter}, {metrics[key]}")

        # also save model to ckpt everytime we run evaluation
        ckpt_file = Path(self.ckpt_dir) / f"ckpt_{iter:06d}.pkl"
        log(f"saving checkpoint to {ckpt_file}")
        with open(ckpt_file, "wb") as f:
            torch.save(ckpt_dict, f)

        ckpt_file = Path(self.ckpt_dir) / "latest.pkl"
        with open(ckpt_file, "wb") as f:
            torch.save(ckpt_dict, f)

    def log_to_wandb(self, metrics: Dict, prefix: str = "", step: int = None):
        if self.wandb_run is not None:
            metrics = gutl.prefix_dict_keys(metrics, prefix=prefix)
            self.wandb_run.log(metrics, step=step)

    @property
    def save_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        return state_dict
