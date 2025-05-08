from data.replay_buffer import ReplayBuffer
from algorithm.evaluator import TaskEvaluator
from gymnasium.wrappers import RecordEpisodeStatistics

import numpy as np
import wandb

class TaskTrainer:
    def __init__(self, env, agent, config):
        self.config = config.algorithm
        self.agent = agent
        self.env = env

        self.buffer_size = config.data.buffer_size
        self.batch_size = config.data.batch_size
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space, self.env.action_space,
            self.buffer_size)

        self.evaluator = TaskEvaluator(env, agent, config)

        self.total_env_steps = 0


    def run_task(self, id, task, hint):
        # wandb metrics - task training
        wandb.define_metric('local_step', overwrite=False)
        wandb.define_metric(f'Training/{task}/*', step_metric='local_step')

        max_steps = self.config.max_steps
        print(f'Learning on task {id}: {task} for {max_steps} steps')
        env = self.env.get_single_env(task)
        env = RecordEpisodeStatistics(env, buffer_length=1) # to record episode return

        prev_success = 0.0
        # reset replay buffer
        self.replay_buffer.reset()

        # start task
        self.agent.start_task(id, hint)

        obs, info = env.reset()  # Reset environment
        for i in range(max_steps):
            # sample action
            if (i < self.config.start_training):
                action = self.env.action_space.sample()
                # initial exploration strategy proposed in ClonEX-SAC
                # if id == 0:
                #     action = self.env.action_space.sample()
                # else:
                #     # choose random previous task (uniform prev strategy)
                #     rand_id = np.random.choice(id)
                #     action = self.agent.sample_action(obs[np.newaxis], rand_id)
                #     action = np.asarray(action, dtype=np.float32).flatten()
            else:
                action = self.agent.sample_actions(obs[np.newaxis], id)
                action = np.asarray(action, dtype=np.float32).flatten()

            # take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # add to buffer
            self.replay_buffer.insert(
                dict(
                    observations=obs,
                    actions=action,
                    rewards=reward,
                    masks=float(not terminated),
                    dones=done,
                    next_observations=next_obs,
                ))

            # repeat or reset environment
            obs = next_obs
            if done:
                # episodic stats for task
                wandb.log({f"Training/{id}-{task}/return": info["episode"]['r']}, commit=False)
                wandb.log({f"Training/{id}-{task}/length": info["episode"]['l']}, commit=False)
                wandb.log({f"Training/{id}-{task}/time": info["episode"]['t']}, commit=False)
                obs, info = env.reset()

            # train
            if (i >= self.config.start_training) and (i % self.config.train_interval == 0):
                for _ in range(self.config.train_updates):
                    batch = self.replay_buffer.sample(self.batch_size)
                    update_info = self.agent.update(batch, id)
                wandb.log({'Agent/actor_loss': update_info['actor_loss'],
                           'Agent/critic_loss': update_info['critic_loss'],
                           'Agent/temperature_loss': update_info['temperature_loss']}, commit=False)

            # evaluate
            if i % self.config.eval_interval == 0:
                stats = self.evaluator.run()
                current_success = stats[f'{id}-{task}/success']
                wandb.log({f'Evaluation/{k}': v for k, v in stats.items()}, commit=False)
                print(f"global step {self.total_env_steps} | local_step {i} | eval: success - {stats['avg_success']}, return - {stats['avg_return']}")

            # logging steps and commit
            wandb.log({
                'global_step': self.total_env_steps,
                'local_step': i
                })

            # increment global step
            self.total_env_steps += 1

            if prev_success >= 0.9 and current_success >= 0.9:
                break
            prev_success = current_success

        # end task
        self.agent.end_task(id)

    def run(self):
        # wandb metrics - evaluator
        wandb.define_metric('global_step')
        wandb.define_metric('Evaluation/*', step_metric='global_step')
        wandb.define_metric('Agent/*', step_metric='global_step')

        for id, task in enumerate(self.env.seq_tasks):
            task, hint = task['task'], task['hint']
            self.run_task(id, task, hint)


