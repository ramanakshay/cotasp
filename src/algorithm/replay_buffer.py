import gymnasium as gym
import numpy as np
import collections

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'next_observations'])

class ReplayBuffer:

    def __init__(self, observation_space, action_space, buffer_size, batch_size):
        self.observations = np.empty((buffer_size, *observation_space.shape),
                                dtype=observation_space.dtype)
        self.actions = np.empty((buffer_size, *action_space.shape),
                           dtype=action_space.dtype)
        self.rewards = np.empty((buffer_size, ), dtype=np.float32)
        # self.masks = np.empty((buffer_size, ), dtype=np.float32)
        self.dones = np.empty((buffer_size, ), dtype=np.float32)
        self.next_observations = np.empty((buffer_size, *observation_space.shape),
                                     dtype=observation_space.dtype)

        self.size = 0
        self.insert_index = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def reset(self):
        # override values
        self.size = 0
        self.insert_index = 0

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, dones: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        # self.masks[self.insert_index] = mask
        self.dones[self.insert_index] = dones
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self) -> Batch:
        indx = np.random.randint(self.size, size=self.batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     # masks=self.masks[indx], # what is mask?
                     next_observations=self.next_observations[indx])