# Continuous Latent Action Models: Towards Capable Robot Policies From Videos

## Installation

```
bash setup.sh
pip3 install -e .
```

## Data Generation

TODO 

## Running CLAM

Sample BC Command:

```
python3 main.py --config-name=train_bc \
    env=metaworld \
    env.dataset_name=tdmpc2-buffer-imgs \
    env.datasets=[mw-assembly] \
    env.env_id=assembly-v2
```

Sample CLAM with joint training training command:

```
python3 main.py --config-name=train_clam \
    env=metaworld \
    env.dataset_name=tdmpc2-buffer-imgs \
    env.datasets=[mw-assembly] \
    env.env_id=assembly-v2 \
    joint_action_decoder_training=True \
    num_labelled_trajs=50 \
    env.action_labelled_dataset=[mw-assembly-al]
```