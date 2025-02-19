import collections
import copy

import einops
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision
import wandb

from clam.models.mlp_policy import MLPPolicy
from clam.models.utils.clam_utils import get_clam_cls, get_la_dim
from clam.trainers.offline_trainer import OfflineTrainer
from clam.utils.data_utils import Batch
from clam.utils.dataloader import get_dataloader
from clam.utils.general_utils import to_device, to_numpy
from clam.utils.logger import log


def get_labelled_dataloader(cfg):
    if cfg.data.labelled_data_type == "trajectory":
        # just get enough trajectories to equate to the same amount of labelled examples
        log("loading expert data for training action decoder", "blue")
        log(f"num trajs to load: {cfg.num_labelled_trajs}")
        cfg.data.num_trajs = cfg.num_labelled_trajs
        cfg.data.num_examples = -1
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    elif cfg.data.labelled_data_type == "transition":
        log("loading random transition data for training action decoder", "blue")
        cfg.data.num_trajs = -1
        cfg.data.num_examples = cfg.num_labelled_trajs * cfg.env.max_episode_steps
        log(f"num examples to load: {cfg.data.num_examples}")
        labelled_dataloader, *_ = get_dataloader(
            cfg=cfg,
            dataset_names=cfg.env.action_labelled_dataset,
            dataset_split=cfg.env.action_labelled_dataset_split,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown labelled data type {cfg.data.labelled_data_type}")

    labelled_dataloader = tf.data.Dataset.sample_from_datasets(
        list(labelled_dataloader.values())
    )
    return labelled_dataloader


class CLAMTrainer(OfflineTrainer):
    def __init__(self, cfg):
        self.use_transformer = "transformer" in cfg.model.idm.net.name
        self.use_soda = "soda" in cfg.model.idm.net.name

        log(f"Using transformer: {self.use_transformer}")
        log(f"Using SODA: {self.use_soda}")

        super().__init__(cfg)
        self.loss_fn = nn.MSELoss(reduction="none")

        if cfg.joint_action_decoder_training:
            # load data for action decoder
            log("loading labelled data for training action decoder")
            cfg_cpy = copy.deepcopy(cfg)
            self.labelled_dataloader = get_labelled_dataloader(cfg_cpy)

            self.labelled_dataloader_train = (
                self.labelled_dataloader.repeat().as_numpy_iterator()
            )

    def setup_action_decoder(self):
        log("---------------------- Initializing Action Decoder ----------------------")

        la_dim = get_la_dim(self.cfg)
        action_decoder = MLPPolicy(
            cfg=self.cfg.model.action_decoder,
            input_dim=la_dim,
            output_dim=self.cfg.env.action_dim,
        )
        action_decoder = action_decoder.to(self.device)
        # self.action_decoder = torch.compile(self.action_decoder)

        log(f"action decoder: {action_decoder}")
        num_params = sum(p.numel() for p in action_decoder.parameters())
        log(f"num params: {num_params}")
        log("=" * 50)
        return action_decoder

    def setup_model(self):
        clam_cls = get_clam_cls(self.cfg.name)
        la_dim = get_la_dim(self.cfg)

        model = clam_cls(self.cfg.model, input_dim=self.obs_shape, la_dim=la_dim)

        if self.cfg.load_from_ckpt:
            cfg, ckpt = model.load_from_ckpt(
                self.cfg.ckpt_file, ckpt_step=self.cfg.ckpt_step, key="model"
            )

        # create action decoder
        if self.cfg.joint_action_decoder_training:
            self.action_decoder = self.setup_action_decoder()
            self.action_decoder_loss_fn = nn.MSELoss(reduction="none")

            if self.cfg.load_from_ckpt:
                cfg, ckpt = self.action_decoder.load_from_ckpt(
                    self.cfg.ckpt_file,
                    ckpt_step=self.cfg.ckpt_step,
                    key="action_decoder",
                )

        return model

    def setup_optimizer_and_scheduler(self):
        opt_cls = getattr(torch.optim, self.cfg.optimizer.name)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.lr_scheduler.name)

        clam_optimizer, clam_scheduler = super().setup_optimizer_and_scheduler()

        if self.cfg.joint_action_decoder_training:
            log("setting up optimizer and scheduler for action decoder", "green")
            self.action_decoder_optimizer = opt_cls(
                self.action_decoder.parameters(), **self.cfg.decoder_optimizer.params
            )
            self.action_decoder_scheduler = scheduler_cls(
                self.action_decoder_optimizer, **self.cfg.lr_scheduler.params
            )

        return clam_optimizer, clam_scheduler

    def compute_action_decoder_loss(self, batch, train: bool = True):
        obs = batch.observations
        k_step = self.cfg.model.fdm.k_step_pred

        if k_step > 1:
            action_preds = []

            for step in range(k_step):
                end = self.cfg.model.context_len + 1 + step
                obs_splice = obs[:, step:end]

                if self.use_transformer:
                    clam_output = self.model(obs_splice, timesteps=batch.timestep)
                else:
                    clam_output = self.model(obs_splice)

                la = clam_output.la

                if self.cfg.model.distributional_la:
                    la = self.model.reparameterize(la)

                action_pred = self.action_decoder(la)
                action_preds.append(action_pred)

            action_pred = torch.stack(action_preds, dim=1)

            if self.use_transformer:
                action_pred = action_pred[:, 0, :-1]
                gt_actions = batch.actions[:, :-1]
            else:
                gt_actions = batch.actions[:, -2 : (-2 + k_step)]  # TODO: fix this
        else:
            # TODO: clean up this logic
            if self.use_transformer:
                # with torch.no_grad():
                clam_output = self.model(
                    obs, timesteps=batch.timestep, states=batch.states
                )
                la = clam_output.la[:, 1:]

                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[:, :-1]
            else:
                clam_output = self.model(obs)
                la = clam_output.la

                action_pred = self.action_decoder(la)
                gt_actions = batch.actions[
                    :, self.cfg.model.context_len :
                ].squeeze()  # TODO: fix this

        assert action_pred.shape == gt_actions.shape, (
            f"{action_pred.shape}, {gt_actions.shape}"
        )

        gt_actions = to_device(gt_actions, self.device)
        action_decoder_loss = self.action_decoder_loss_fn(action_pred, gt_actions)
        action_decoder_loss = action_decoder_loss.mean()
        return action_decoder_loss, {"action_decoder_loss": action_decoder_loss.item()}

    def compute_loss(self, batch, train: bool = True):
        """
        Compute loss for the Latent Action Model.

        If we are doing k-step prediction,
        we use the reconstructed observation for the next steps prediction.

        e.g. If k_step = 2, we predict o_t+1 and o_t+2

        Step 1:
            o_t-1, o_t => o^_t+1 and z_t
        Step 2:
            o_t, o^_t+1 => o^_t+2 and z_t+1

        Then we compute the summed loss for both o^_t+1 and o^_t+2.
        """
        k_step = self.cfg.model.fdm.k_step_pred

        # check if we are doing k-step prediction
        if k_step > 1:
            # recursively predict the next obs and latent actions
            def splice(x, start, end):
                if x is not None:
                    return x[:, start:end]
                return x

            total_loss = 0.0
            obs_recons = []
            metrics = collections.defaultdict(float)

            for step in range(k_step):
                # need to splice Batch
                end = self.cfg.model.context_len + 1 + step
                batch_splice = dict(
                    map(
                        lambda kv: (kv[0], splice(kv[1], start=step, end=end)),
                        batch.__dict__.items(),
                    )
                )
                batch_splice = Batch(**batch_splice)

                # replace the second to last observation with the reconstructed observation
                if len(obs_recons) > 0:
                    new_observations = batch_splice.observations.clone()
                    new_observations[:, -2] = obs_recons[-1]
                    batch_splice.observations = new_observations

                # compute LAM loss for single step
                step_metrics, obs_recon, step_loss = self.compute_step_loss(
                    batch_splice, train=train
                )

                for k, v in step_metrics.items():
                    metrics[k] += v

                obs_recons.append(obs_recon)
                total_loss += step_loss

            obs_recons = torch.stack(obs_recons, dim=-1)
            metrics = {k: v / k_step for k, v in metrics.items()}
        else:
            metrics, obs_recon, total_loss = self.compute_step_loss(batch, train=train)

        return metrics, total_loss

    def compute_step_loss(self, batch, train: bool = True):
        """
        Computes IDM/FDM loss for a single step.

        e.g. o_t-1, o_t => o_t+1, we also predict z_t
        which is the latent action between o_t and o_t+1
        """
        # reconstructs full sequence if using a transformer model
        if self.use_transformer:
            clam_output = self.model(
                batch.observations, timesteps=batch.timestep, states=batch.states
            )
            obs_gt = batch.observations[:, 1:]

            cheat_pred = batch.observations[:, :-1]  # this refers to o_t
            cheat_loss = self.loss_fn(cheat_pred, obs_gt)
            cheat_loss = cheat_loss.mean()
        else:
            clam_output = self.model(batch.observations)

            if self.cfg.model.fdm.predict_target_embedding:
                # compute target embedding for the next observation
                pass
            else:
                # reconstruct the next raw observation
                obs_gt = batch.observations[:, self.cfg.model.context_len :].squeeze()

            # compute cheating loss, this is if we just predict the current obs as the next obs
            cheat_pred = batch.observations[
                :, self.cfg.model.context_len - 1 : -1
            ].squeeze()  # this refers to o_t
            cheat_loss = self.loss_fn(cheat_pred, obs_gt)
            cheat_loss = cheat_loss.mean()

        obs_recon = clam_output.reconstructed_obs
        la = clam_output.la

        # make sure the shapes are correct
        assert obs_recon.shape == obs_gt.shape, f"{obs_recon.shape}, {obs_gt.shape}"

        recon_loss = self.loss_fn(obs_recon, obs_gt)
        recon_loss = recon_loss.mean()
        total_loss = self.cfg.model.recon_loss_weight * recon_loss

        # compute KL divergence loss to regularize the latent action
        if self.cfg.model.distributional_la and self.cfg.model.kl_loss_weight:
            la_mean = la[:, : self.cfg.model.la_dim]
            la_logvar = la[:, self.cfg.model.la_dim :]

            posterior = torch.distributions.Normal(la_mean, torch.exp(0.5 * la_logvar))
            prior = torch.distributions.Normal(0, 1)
            kl_loss = torch.distributions.kl.kl_divergence(posterior, prior)
            kl_loss = kl_loss.mean()
            total_loss += self.cfg.model.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        metrics = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "la_mean": la.mean().item(),
            "la_min": la.min().item(),
            "la_max": la.max().item(),
            "obs_recon_mean": obs_recon.mean().item(),
            "obs_recon_min": obs_recon.min().item(),
            "obs_recon_max": obs_recon.max().item(),
            "cheat_loss": cheat_loss.item(),
        }

        if self.cfg.model.idm.quantize_la:
            if clam_output.idm_output.vq_loss is not None:
                total_loss += clam_output.idm_output.vq_loss

            metrics.update(clam_output.idm_output.vq_metrics)

        # run this if we are doing joint training or if this is eval
        if (
            self.cfg.joint_action_decoder_training
            and self.train_step % self.cfg.train_action_decoder_every == 0
            and train
        ):
            labelled_batch = next(self.labelled_dataloader_train)
            labelled_batch = to_device(labelled_batch, self.device)
            labelled_batch = Batch(**labelled_batch)

            action_decoder_loss, action_decoder_metrics = (
                self.compute_action_decoder_loss(labelled_batch)
            )
            total_loss += self.cfg.action_decoder_loss_weight * action_decoder_loss
            metrics.update(action_decoder_metrics)

        return metrics, obs_recon, total_loss

    @property
    def save_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.cfg.joint_action_decoder_training:
            state_dict["action_decoder"] = self.action_decoder.state_dict()
            state_dict["action_decoder_opt"] = (
                self.action_decoder_optimizer.state_dict()
            )
        return state_dict

    def eval(self, step: int):
        super().eval(step=step)

        if self.cfg.joint_action_decoder_training:
            log("running action decoder evaluation", "blue")

            self.model.eval()
            self.action_decoder.eval()

            # first run on eval set
            eval_iter = self.eval_dataloader.as_numpy_iterator()

            eval_metrics = collections.defaultdict(list)
            for batch in eval_iter:
                batch = to_device(batch, self.device)
                batch = Batch(**batch)

                with torch.no_grad():
                    action_decoder_loss, action_decoder_metrics = (
                        self.compute_action_decoder_loss(batch)
                    )

                    for k, v in action_decoder_metrics.items():
                        eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(np.array(v))

            self.log_to_wandb(eval_metrics, prefix="action_decoder_eval/")

            # then run on the full labelled dataset
            eval_metrics = collections.defaultdict(list)
            labelled_dataloader_eval = self.labelled_dataloader.as_numpy_iterator()

            for batch in labelled_dataloader_eval:
                batch = to_device(batch, self.device)
                batch = Batch(**batch)

                with torch.no_grad():
                    action_decoder_loss, action_decoder_metrics = (
                        self.compute_action_decoder_loss(batch)
                    )

                    for k, v in action_decoder_metrics.items():
                        eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(np.array(v))

            self.log_to_wandb(eval_metrics, prefix="action_decoder_labelled_eval/")

        if self.cfg.env.image_obs and not self.cfg.model.fdm.predict_target_embedding:
            # visualize the image reconstructions on eval set
            log("visualizing image reconstructions", "blue")

            # sample a batch
            eval_iter = self.eval_dataloader.as_numpy_iterator()
            batch = next(eval_iter)
            batch = to_device(batch, self.device)
            batch = Batch(**batch)

            if self.use_transformer:
                clam_output = self.model(
                    batch.observations, timesteps=batch.timestep, states=batch.states
                )
                obs_recon = clam_output.reconstructed_obs[:, :-1]
                obs_gt = batch.observations[:, 1:]

                # we will need to flatten the first two dimensions
                obs_recon = einops.rearrange(obs_recon, "B T C H W -> (B T) C H W")
                obs_gt = einops.rearrange(obs_gt, "B T C H W -> (B T) C H W")
            else:
                clam_output = self.model(batch.observations)
                obs_recon = clam_output.reconstructed_obs
                obs_gt = batch.observations[:, -1]

            # wandb expects [B, H, W, C]
            assert obs_recon.ndim == 4

            num_exs = 5
            obs_recon = obs_recon[:num_exs]
            obs_gt = obs_gt[:num_exs]

            # if there is framestack, take the last frame
            if self.cfg.env.n_frame_stack > 1:
                obs_recon = obs_recon[:, -3:]
                obs_gt = obs_gt[:, -3:]

            # mix them together
            to_vis = torch.cat([obs_recon, obs_gt])
            to_vis = torchvision.utils.make_grid(to_vis, nrow=num_exs)
            # make channel last
            to_vis = einops.rearrange(to_vis, "c h w -> h w c")
            to_vis = to_numpy(to_vis)

            # plot images
            self.log_to_wandb({"obs_recon_1": wandb.Image(to_vis)}, prefix="images/")
