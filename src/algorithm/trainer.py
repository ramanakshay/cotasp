from algorithm.replay_buffer import ReplayBuffer
from algorithm.evaluator import TaskEvaluator

import numpy as np
import wandb

class TaskTrainer:
    def __init__(self, env, agent, config):
        self.config = config.algorithm
        self.agent = agent
        self.env = env

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space, self.env.action_space,
            self.config.buffer_size, self.config.batch_size
        )

        self.evaluator = TaskEvaluator(env, agent, config)

        self.total_env_steps = 0


    def run_task(self, id, task, hint):
        # wandb metrics
        wandb.define_metric('local_step', overwrite=True)
        wandb.define_metric(f'Training/{task}/*', step_metric='local_step')

        max_steps = self.config.max_steps
        print(f'Learning on task {id}: {task} for {max_steps} steps')
        env = self.env.get_single_env(task)
        self.replay_buffer.reset()

        # start task
        self.agent.start_task(id, hint)

        obs, info = env.reset()  # Reset environment
        eps_r = 0.0
        for i in range(max_steps):
            # sample action
            if (i < self.config.start_training):
                # initial exploration strategy proposed in ClonEX-SAC
                if id == 0:
                    action = self.env.action_space.sample()
                else:
                    # choose random previous task (uniform prev strategy)
                    rand_id = np.random.choice(id)
                    action = self.agent.sample_action(obs, rand_id)
                    action = np.asarray(action, dtype=np.float32).flatten()
            else:
                action = self.agent.sample_action(obs, id)
                action = np.asarray(action, dtype=np.float32).flatten()

            # take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            eps_r += reward
            done = terminated or truncated

            # add to buffer
            self.replay_buffer.insert(obs, action, reward, float(done), next_obs)

            # repeat
            obs = next_obs
            if done:
                # episodic returns for task
                wandb.log({'local_step': i}, commit=False)
                wandb.log({f'Training/{task}/eps_r': eps_r})
                obs, info = env.reset()
                eps_r = 0.0

            # train
            if (i >= self.config.start_training) and (i % self.config.train_interval == 0):
                for _ in range(self.config.train_updates):
                    batch = self.replay_buffer.sample()
                    update_info = self.agent.update(batch, id)

            # evaluate
            if i % self.config.eval_interval == 0:
                stats = self.evaluator.run()
                wandb.log({'global_step': self.total_env_steps}, commit=False)
                wandb.log({f'Evaluation/{k}': v for k, v in stats.items()})
                print(f"global step {self.total_env_steps} | local_step {i} | eval: success - {stats['avg_success']}, return - {stats['avg_return']}")

            # increment global step
            self.total_env_steps += 1

        # end task
        self.agent.end_task(id)

    def run(self):
        wandb.define_metric('global_step')
        wandb.define_metric('Evaluation/*', step_metric='global_step')
        for id, task in enumerate(self.env.seq_tasks):
            task, hint = task['task'], task['hint']
            self.run_task(id, task, hint)


