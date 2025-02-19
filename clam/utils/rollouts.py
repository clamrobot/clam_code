import collections
import copy
import time
from typing import Dict, List, Tuple, Union

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import tqdm
from matplotlib import cm, font_manager
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

import wandb
from clam.envs.utils import make_envs, procgen_action_meanings
from clam.utils.data_utils import Transition
from clam.utils.general_utils import to_numpy
from clam.utils.logger import log


def annotate_video(
    video: np.ndarray,
    annotations: Dict,
    img_size: Tuple[int, int] = (256, 256),
):
    # load a nice big readable font
    font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(font)
    font = ImageFont.truetype(file, 12)

    annotated_imgs = []
    for step, frame in enumerate(video):
        frame = Image.fromarray(frame)
        frame = frame.resize(img_size)

        # add border on top of image
        extra_border_height = 100
        annotated_img = Image.new(
            "RGB",
            (frame.width, frame.height + extra_border_height),
            color=(255, 255, 255),
        )
        annotated_img.paste(frame, (0, extra_border_height))
        draw = ImageDraw.Draw(annotated_img)

        count = 0
        lines = []
        to_display = ""
        num_keys_per_line = 2

        for key, values in annotations.items():
            if isinstance(values[step], np.ndarray):
                values = np.round(values[step], 3)
                to_add = f"{key}: {values}  "
            elif isinstance(values[step], float) or isinstance(
                values[step], np.float32
            ):
                to_add = f"{key}: {values[step]:.3f}  "
            else:
                to_add = f"{key}: {values[step]}  "

            if count < num_keys_per_line:
                to_display += to_add
                count += 1
            else:
                lines.append(to_display)
                count = 1
                to_display = to_add

        # add the last line
        if lines:
            lines.append(to_display)

        for i, line in enumerate(lines):
            # make font size bigger
            draw.text((10, 10 + i * 20), line, fill="black", font=font)

        # convert to numpy array
        annotated_imgs.append(np.array(annotated_img))

    return np.array(annotated_imgs)


def run_eval_rollouts(cfg: DictConfig, model: nn.Module, wandb_run=None):
    """
    Run evaluation rollouts
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_time = time.time()
    cfg.env.num_envs = cfg.num_eval_rollouts

    envs = make_envs(training=False, **cfg.env)
    eval_metrics, rollouts = rollout_helper(
        cfg, envs, model=model, log_videos=False, device=device
    )
    
    envs.close()

    rollout_time = time.time() - rollout_time

    eval_metrics["time"] = rollout_time / cfg.num_eval_rollouts

    # compute normalized returns if d4rl mujoco env
    if cfg.env.env_name == "mujoco":
        # create a dummy env
        env_cfg = copy.deepcopy(cfg.env)
        env_cfg.num_envs = 1
        dummy_env = make_envs(training=False, **env_cfg)
        normalized_return = dummy_env.envs[0].get_normalized_score(
            eval_metrics["ep_ret"]
        )
        normalized_return_std = dummy_env.envs[0].get_normalized_score(
            eval_metrics["ep_ret_std"]
        )
        eval_metrics["normalized_ret"] = normalized_return
        eval_metrics["normalized_ret_stderr"] = normalized_return_std / np.sqrt(
            cfg.num_eval_rollouts
        )

    # for some environments, we need to run it sequentially to generate the videos
    # this is just for the rendering rollout videos
    if cfg.log_rollout_videos:
        cfg = copy.deepcopy(cfg)
        cfg.env.num_envs = 1
        video_env = make_envs(training=False, **cfg.env)

        rollout_videos = []
        for _ in tqdm.tqdm(
            range(cfg.num_eval_rollouts_render), desc="generating videos"
        ):
            _, rollouts = rollout_helper(
                cfg, video_env, model, log_videos=cfg.log_rollout_videos, device=device
            )

            # squeeze the N envs dimension
            info = rollouts["info"]
            rollouts = {k: v.squeeze() for k, v in rollouts.items() if k != "info"}
            rollouts["info"] = info
            num_steps = rollouts["observation"].shape[0]
            ret = rollouts["reward"].cumsum(axis=0)

            annotations = {
                "S": np.arange(num_steps),
                "Ret": ret,
                "R": rollouts["reward"],
                "D": rollouts["done"],
                "A": rollouts["action"],
            }

            # select keys from the environment infos to log in our rollout video
            if cfg.env.env_name == "metaworld":
                env_ann = {
                    "Grasp": rollouts["info"]["grasp_success"].squeeze().astype(int),
                    "Dist": rollouts["info"]["obj_to_target"].squeeze(),
                    "SUC": rollouts["info"]["success"].squeeze().astype(int),
                }
            elif cfg.env.env_name == "mujoco":
                env_ann = {
                    "Rew_ctrl": rollouts["info"]["reward_ctrl"].squeeze(),
                    "Rew_run": rollouts["info"]["reward_run"].squeeze(),
                }
            else:
                env_ann = {}

            annotations.update(env_ann)
            annotated_video = annotate_video(rollouts["image"], annotations)
            rollout_videos.append(annotated_video)
            
        video_env.close()

        rollout_videos = np.array(rollout_videos)

        if wandb_run is not None:
            rollout_videos = einops.rearrange(rollout_videos, "n t h w c -> n t c h w")
            wandb_run.log(
                {"rollout_videos/": wandb.Video(rollout_videos, fps=cfg.video_fps)}
            )

    return eval_metrics, rollouts


def rollout_helper(
    cfg: DictConfig,
    env,
    model: nn.Module,
    log_videos: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Union[Dict[str, np.ndarray], List[Transition]]:
    """
    Return:
        eval_metrics: Dict of evaluation metrics
        transitions: List of transitions
    """
    # gym version needs to be gym==0.23.1 for this to work
    log(f"running rollouts, log videos: {log_videos}", color="green")

    obs = env.reset()
    if cfg.env.env_name == "metaworld":
        obs, info = obs

    n_envs = obs.shape[0]

    curr_timestep = 0
    dones = np.zeros((cfg.num_eval_rollouts,))
    ep_returns = np.zeros((cfg.num_eval_rollouts,))
    ep_lengths = np.zeros((cfg.num_eval_rollouts,))
    ep_success = np.zeros((cfg.num_eval_rollouts,))

    rollouts = collections.defaultdict(list)

    while not np.all(dones):
        # break after max timesteps
        if curr_timestep >= cfg.env.max_episode_steps:
            break

        obs = torch.from_numpy(obs).to(device).float()

        # normalize if using image observations
        if cfg.env.image_obs and not cfg.env.use_pretrained_embeddings:
            obs = obs / 255.0
            # also reshape so channel comes first
            obs = obs.permute(0, 3, 1, 2)

        # run forward pass to get the action
        if cfg.model.gaussian_policy:
            # for now make the action deterministic
            action_mean, action_log_std = model(obs)
            action_pred = action_mean
        else:
            action_pred = model(obs)
        action = to_numpy(action_pred)

        # step in the environment
        if cfg.env.env_name == "metaworld":
            obs, reward, done, _, info = env.step(action)
        else:
            obs, reward, done, info = env.step(action)

        # update episode returns and lengths
        dones = np.logical_or(dones, done)
        ep_rew = reward * (1 - dones)
        step = np.ones_like(dones) * (1 - dones)
        ep_returns += ep_rew
        ep_lengths += step

        if cfg.env.env_name == "metaworld":
            if "final_info" in info:
                info = rollouts["info"][-1]  # use the last info
            else:
                ep_success = np.logical_or(ep_success, info["success"])
        else:
            # this is for CALVIN
            success = [info[i]["success"] for i in range(n_envs)]
            ep_success = np.logical_or(ep_success, success)

        # generate image frames for the video
        if log_videos:
            if cfg.env.env_name == "procgen":
                image = obs
            else:
                if cfg.env.env_name == "metaworld":
                    image = env.call("render")
                else:
                    # need to render each env separately
                    image = env.call("render", mode="rgb_array")
                # tuple to array
                image = np.array(image)

                # if cfg.env.env_name == "metaworld":
                #     # flip the image vertically
                #     # test
                #     image[0] = np.flipud(image[0])
        else:
            image = None

        # fix the last info timestep for mujoco hc because DMC env returns
        # extra keys in the info dict
        if cfg.env.env_name == "mujoco" and "terminal_observation" in info[0]:
            for i in range(n_envs):
                info[i].pop("terminal_observation")
                info[i].pop("TimeLimit.truncated")

        rollouts["observation"].append(obs)
        rollouts["action"].append(action)
        rollouts["reward"].append(reward)
        rollouts["done"].append(done)
        rollouts["image"].append(image)
        rollouts["info"].append(info)
        curr_timestep += 1

    # convert to numpy arrays
    for k, v in rollouts.items():
        if not isinstance(v[0], dict) and v[0] is not None:
            rollouts[k] = np.array(v)
            rollouts[k] = np.swapaxes(rollouts[k], 0, 1)

    # infos is a list of dicts
    infos = rollouts["info"]

    if cfg.env.env_name == "metaworld":
        # flatten the timesteps
        infos = {k: np.array([info[k] for info in infos]) for k in infos[0].keys()}

    else:
        # first flatten the envs
        infos_env = []
        for step in infos:
            _infos_env = {
                k: np.array([info[k] for info in step]) for k in step[0].keys()
            }
            infos_env.append(_infos_env)

        # then flatten to timestep
        infos = {
            k: np.array([info[k] for info in infos_env]) for k in infos_env[0].keys()
        }
    rollouts["info"] = infos

    eval_metrics = {
        "ep_ret": np.mean(ep_returns),
        "ep_ret_std": np.std(ep_returns),
        "avg_len": np.mean(ep_lengths),
        "std_len": np.std(ep_lengths),
        "max_len": np.max(ep_lengths),
        "min_len": np.min(ep_lengths),
        "success_rate": np.mean(ep_success),
    }
    return eval_metrics, rollouts
