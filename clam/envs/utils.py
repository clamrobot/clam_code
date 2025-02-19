from functools import partial

import d4rl
import numpy as np
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# from procgen import ProcgenEnv
from clam.envs.wrappers import (
    ActionRepeatWrapper,
    FlipImageObsWrapper,
    FrameStackWrapper,
    ImageObservationWrapper,
    R3MWrapper,
)

ALL_PROCGEN_GAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]

# def normalize_return(ep_ret, env_name):
#     """normalizes returns based on URP and expert returns above"""
#     return doy.normalize_into_range(
#         lower=urp_ep_return[env_name],
#         upper=expert_ep_return[env_name],
#         v=ep_ret,
#     )

procgen_action_meanings = np.array(
    [
        "LEFT-DOWN",
        "LEFT",
        "LEFT-UP",
        "DOWN",
        "NOOP",
        "UP",
        "RIGHT-DOWN",
        "RIGHT",
        "RIGHT-UP",
        "D",
        "A",
        "W",
        "S",
        "Q",
        "E",
    ]
)


# taken from: https://github.com/schmidtdominik/LAPO/blob/main/lapo/env_utils.py
def make_procgen_envs(num_envs, env_id, gamma, **kwargs):
    import gym

    envs = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_id,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
        rand_seed=0,  # TODO: set this
    )

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    assert isinstance(envs.action_space, gym.spaces.discrete.Discrete), (
        "only discrete action space is supported"
    )

    # envs.normalize_return = partial(normalize_return, env_name=env_id)
    return envs


def make_mujoco_envs(num_envs, env_id, **kwargs):
    import gym

    # wrap multienv
    # envs = gym.vector.make(env_id, num_envs=num_envs)

    # use sync vector env
    env_fn = lambda: gym.make(env_id)
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv([env_fn])
    else:
        envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])
    return envs


def make_metaworld_envs(num_envs, env_id, **kwargs):
    import gymnasium as gym
    import metaworld

    mw_env_id = env_id + "-goal-observable"
    if mw_env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError("Unknown task:", mw_env_id)

    env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[mw_env_id]
    # ml1 = metaworld.ML1(env_id)
    # env_cls = ml1.train_classes[env_id]

    def env_fn(env_idx):
        # this is required to ensure that each environment
        # has a different random seed
        st0 = np.random.get_state()
        np.random.seed(env_idx)

        # env = env_cls(render_mode="rgb_array", camera_name="corner2")

        env = env_cls(seed=env_idx, render_mode="rgb_array")
        # env.camera_name = "corner2"
        # env.model.cam_pos[2] = [0.75, 0.075, 0.7]

        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env._freeze_rand_vec = True
        env.seed(env_idx)
        np.random.set_state(st0)

        # add MW wrappers if we are using TDMPC2 code to change camera angle
        # env = TDMPC2MWWrapper(env, image_shape=kwargs["image_shape"])

        if kwargs["image_obs"]:
            env = ImageObservationWrapper(env, image_shape=kwargs["image_shape"])

            # flip the image, this is only for visualization...
            # env = FlipImageObsWrapper(env)

            if kwargs["use_pretrained_embeddings"]:
                env = R3MWrapper(env, r3m_id=kwargs["r3m_id"])

            if kwargs["n_frame_stack"] > 1:
                env = FrameStackWrapper(env, num_frames=kwargs["n_frame_stack"])

        if kwargs["action_repeat"] > 1:
            env = ActionRepeatWrapper(env, kwargs["action_repeat"])

        return env

    envs = [partial(env_fn, env_idx=i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(envs)
    else:
        envs = gym.vector.AsyncVectorEnv(envs)
    return envs


def make_calvin_envs(num_envs, env_id, **kwargs):
    import gym
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

    from clam.envs.calvin_env import SlideEnv

    # load config from yaml file
    with open("calvin_env.yaml", "r") as stream:
        calvin_cfg = OmegaConf.load(stream)

    def env_fn(env_idx):
        env = SlideEnv(seed=env_idx, target_task=env_id, **calvin_cfg)
        return env

    envs = [partial(env_fn, env_idx=i) for i in range(num_envs)]
    if num_envs == 1:
        envs = gym.vector.SyncVectorEnv(envs)
    else:
        envs = SubprocVecEnv(envs, start_method="fork")
        envs.render_mode = "rgb_array"

    return envs


def make_envs(env_name: str, num_envs: int, env_id: str, **kwargs):
    if env_name == "procgen":
        envs = make_procgen_envs(num_envs=num_envs, env_id=env_id, gamma=1.0)
    elif env_name == "mujoco":
        envs = make_mujoco_envs(num_envs=num_envs, env_id=env_id, **kwargs)
    elif env_name == "metaworld":
        envs = make_metaworld_envs(num_envs=num_envs, env_id=env_id, **kwargs)
    elif env_name == "calvin":
        envs = make_calvin_envs(num_envs=num_envs, env_id=env_id, **kwargs)

    return envs
