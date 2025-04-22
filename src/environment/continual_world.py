from typing import List
import random
import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformReward, ClipAction

def get_mt50() -> metaworld.MT50:
    saved_random_state = np.random.get_state()
    np.random.seed(999)
    random.seed(999)
    MT50 = metaworld.MT50()
    np.random.set_state(saved_random_state)
    return MT50

TASK_SEQS = {
    "cw10": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "push-wall-v2", 'hint': 'Bypass a wall and push a puck to a goal.'},
        {'task': "faucet-close-v2", 'hint': 'Rotate the faucet clockwise.'},
        {'task': "push-back-v2", 'hint': 'Pull a puck to a goal.'},
        {'task': "stick-pull-v2", 'hint': 'Grasp a stick and pull a box with the stick.'},
        {'task': "handle-press-side-v2", 'hint': 'Press a handle down sideways.'},
        {'task': "push-v2", 'hint': 'Push the puck to a goal.'},
        {'task': "shelf-place-v2", 'hint': 'Pick and place a puck onto a shelf.'},
        {'task': "window-close-v2", 'hint': 'Push and close a window.'},
        {'task': "peg-unplug-side-v2", 'hint': 'Unplug a peg sideways.'},
    ],
    "cw1-hammer": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
    ],
    "cw1-push-back": [
        {'task': "push-back-v2", 'hint': 'Pull a puck to a goal.'},
    ],
    "cw1-push": [
        {'task': "push-v2", 'hint': 'Push the puck to a goal.'},
    ],
    "cw2-test": [
        {'task': "push-wall-v2", 'hint': 'Bypass a wall and push a puck to a goal.'},
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
    ],
    "cw2-ab-coffee-button": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "coffee-button-v2", 'hint': 'Push a button on the coffee machine.'}
    ],
    "cw2-ab-handle-press": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "handle-press-v2", 'hint': 'Press a handle down.'}
    ],
    "cw2-ab-window-open": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "window-open-v2", 'hint': 'Push and open a window.'}
    ],
    "cw2-ab-reach": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "reach-v2", 'hint': 'Reach a goal position.'}
    ],
    "cw2-ab-button-press": [
        {'task': "hammer-v2", 'hint': 'Hammer a screw on the wall.'},
        {'task': "button-press-v2", 'hint': 'Press a button.'}
    ],
    "cw3-test": [
        {'task': "stick-pull-v2", 'hint': 'Grasp a stick and pull a box with the stick.'},
        {'task': "push-back-v2", 'hint': 'Pull a puck to a goal.'},
        {'task': "shelf-place-v2", 'hint': 'Pick and place a puck onto a shelf.'},
    ]
}

TASK_SEQS["cw20"] = TASK_SEQS["cw10"] + TASK_SEQS["cw10"]
META_WORLD_TIME_HORIZON = 200
MT50 = get_mt50()

class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind

        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False
            env.seeded_rand_vec = True

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> np.ndarray:
        if self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec
        return self.env.reset(**kwargs)

class ContinualWorld:
    def __init__(self, config):
        self.config = config.environment
        self.seq = self.config.seq
        self.randomization = self.config.randomization
        self.normalize_reward = self.config.normalize_reward
        self.seed = config.system.seed + 42
        self.seq_tasks = TASK_SEQS[self.seq]
        self.num_tasks = len(self.seq_tasks)
        
        # same observation and action spaces for all tasks
        env = self.get_single_env('pick-place-v2')  # sample any environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env # delete temporary environment


    def get_subtasks(self, name: str) -> List[metaworld.Task]:
        # TODO: what are subtasks?
        return [s for s in MT50.train_tasks if s.env_name == name]

    def get_single_env(self, name):
        if name == "HalfCheetah-v3" or name == "Ant-v3":
            env = gym.make(name)
            env.seed(self.seed)
            env.action_space.seed(self.seed)
            env.observation_space.seed(self.seed)
        else:
            env = MT50.train_classes[name]()
            env.seed(self.seed)
            env = RandomizationWrapper(env, self.get_subtasks(name), self.randomization)
            env = TimeLimit(env, META_WORLD_TIME_HORIZON)
        env = ClipAction(env)
        if self.normalize_reward:
            env = TransformReward(env, lambda r: r / META_WORLD_TIME_HORIZON)
        env.name = name
        return env


if __name__ == "__main__":
    import time

    # def print_reward(env: gym.Env):
    #     obs, done = env.reset(), False
    #     i = 0
    #     while not done:
    #         i += 1
    #         next_obs, rew, done, _ = env.step(env.action_space.sample())
    #         print(i, rew)

    # tasks_list = TASK_SEQS["cw1-push"]
    # env = get_single_env(tasks_list[0], 1, "deterministic", normalize_reward=False)
    # env_normalized = get_single_env(tasks_list[0], 1, "deterministic", normalize_reward=True)

    # print_reward(env)
    # print_reward(env_normalized)

    # ALL tasks
    print(metaworld.MT50.train_tasks)

    tasks_list = TASK_SEQS["cw1-push"]
    cw = ContinualWorldEnvironment()

    s = time.time()
    env = cw.get_single_env(tasks_list[0], 1, "random_init_all")
    print(time.time() - s)
    s = time.time()
    env = cw.get_single_env(tasks_list[0], 1, "random_init_all")
    print(time.time() - s)

    o, i = env.reset()
    _, _, _, _, _ = env.step(np.array([np.nan, 1.0, -1.0, 0.0]))
    o_new, i = env.reset()
    print(o)
    print(o_new)