from environment import ContinualWorld
from algorithm import TaskTrainer
from agent.random import RandomAgent

import gymnasium as gym
import random
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf


def setup(config):
    np.random.seed(config.system.seed)
    random.seed(config.system.seed)

def cleanup():
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    setup(config)

    ## ENVIRONMENT ##
    env = ContinualWorld(config)
    print("Environment Loaded.")

    ## AGENT ##
    agent = RandomAgent(env.observation_space, env.action_space, config)
    print('Agent Created.')

    ## ALGORITHM ##
    print('Algorithm Running.')
    alg = TaskTrainer(env, agent, config)
    alg.run()
    print('Done!')

    ## CLEANUP ##
    cleanup()

if __name__ == "__main__":
    main()