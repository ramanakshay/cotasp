import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

class TaskEvaluator:
    def __init__(self, env, agent, config):
        '''
        Env should be of kind seq-task
        '''
        self.config = config.algorithm
        self.agent = agent
        self.env = env

        eval_envs = []
        for id, task in enumerate(self.env.seq_tasks):
            task = task['task']
            eval_envs.append(self.env.get_single_env(task))
        self.eval_envs = eval_envs

    def run(self):
        stats = {}
        sum_return = 0.0
        sum_success = 0.0
        list_log_keys = ['r']

        # dummy inputs
        # dummy_obs = jnp.ones((128, 12))

        for id, env in enumerate(self.eval_envs):
            # record statistics
            name = env.name
            env = RecordEpisodeStatistics(env, buffer_length=self.config.eval_episodes)
            for k in list_log_keys:
                stats[f'{id}-{name}/{k}'] = []
            successes = None

            for _ in range(self.config.eval_episodes):
                # run episode
                done = False
                obs, info = env.reset()
                while not done:
                    action = self.agent.sample_actions(obs, id)
                    action = np.asarray(action, dtype=np.float32).flatten()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                for k in list_log_keys:
                    stats[f'{id}-{name}/{k}'].append(info['episode'][k])

                if 'success' in info:
                    if successes is None:
                        successes = 0.0
                    successes += info['success']

            for k in list_log_keys:
                stats[f'{id}-{name}/{k}'] = np.mean(stats[f'{id}-{name}/{k}'])

            if successes is not None:
                stats[f'{id}-{name}/success'] = successes / self.config.eval_episodes
                sum_success += stats[f'{id}-{name}/success']

            sum_return += stats[f'{id}-{name}/r']

            # stats[f'{task_i}-{env.name}/check_dummy_action'] = agent.sample_actions(dummy_obs, task_i, temperature=0).mean()

        stats['avg_return'] = sum_return / len(self.eval_envs)
        stats['avg_success'] = sum_success / len(self.eval_envs)

        return stats

