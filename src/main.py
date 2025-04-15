from environment import ContinualWorld
from algorithm import TaskTrainer
import gymnasium as gym

import hydra
from omegaconf import DictConfig, OmegaConf


def setup():
    pass

def cleanup():
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##

    ## ENVIRONMENT ##
    env = ContinualWorld(config)
    print("Environment Loaded.")

    # ## AGENT ##
    agent = None
    # agent = DiscreteActorCritic(config)
    # print('Agent Created.')

    # ## ALGORITHM ##
    print('Algorithm Running.')
    alg = TaskTrainer(agent, env, config)
    alg.run()
    print('Done!')

    ## CLEANUP ##

if __name__ == "__main__":
    main()