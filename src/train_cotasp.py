from environment import ContinualWorld
from algorithm import TaskTrainer
from agent.cotasp import CoTASPAgent

import gymnasium as gym
import random
import numpy as np
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf


def setup(config):
    # set random seed
    np.random.seed(config.system.seed)
    random.seed(config.system.seed)

    # wandb init
    wandb.init(
        entity=config.wandb.entity,
        project=config.wandb.project,
        name=config.wandb.name,
        config=OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        ),
        mode=config.wandb.mode,
        settings=wandb.Settings(start_method="thread")
    )

def cleanup():
    wandb.finish()

@hydra.main(version_base=None, config_path="config", config_name="train_cotasp")
def main(config : DictConfig) -> None:
    ## SETUP ##
    setup(config)

    ## ENVIRONMENT ##
    env = ContinualWorld(config)
    print("Environment Loaded.")

    ## AGENT ##
    agent = CoTASPAgent(env.observation_space, env.action_space, env.num_tasks, config)
    print('Agent Created.')

    ## ALGORITHM ##
    # print('Algorithm Running.')
    # alg = TaskTrainer(env, agent, config)
    # alg.run()
    # print('Done!')

    ## CLEANUP ##
    cleanup()

if __name__ == "__main__":
    main()