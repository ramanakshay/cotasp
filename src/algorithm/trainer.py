from algorithm.replay_buffer import ReplayBuffer

class TaskTrainer:
    def __init__(self, env, agent, config):
        self.config = config.algorithm
        self.agent = agent
        self.env = env

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space, self.env.action_space,
            self.config.buffer_size, self.config.batch_size
        )

        eval_envs = []
        for id, task in enumerate(self.env.seq_tasks):
            task = task['task']
            eval_envs.append(self.env.get_single_env(task))
        self.eval_envs = eval_envs

        self.total_env_steps = 0

    def train(self, id):
        for _ in range(self.config.train_updates):
            batch = self.replay_buffer.sample()
            update_info = self.agent.update(id, batch)

    def evaluate(self):
        pass

    def run_task(self, id, task, hint):
        max_steps = self.config.max_steps
        print(f'Learning on task {id}: {task} for {max_steps} steps')
        env = self.env.get_single_env(task)
        self.replay_buffer.reset()

        # start task
        self.agent.start_task(id, hint)

        obs, info = env.reset()  # Reset environment
        for i in range(max_steps):
            # sample action
            act = self.agent.sample_action(id, obs)  # Sample an action
            # take step
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            # add to buffer
            self.replay_buffer.insert(obs, act, reward, float(done), next_obs)
            # repeat
            obs = next_obs
            if done:
                obs, info = env.reset()
            # train
            if (i >= self.config.start_training) and (i % self.config.train_interval == 0):
                self.train(id)
            #eval
            if i % self.config.eval_interval == 0:
                self.evaluate()
            # increment global step
            self.total_env_steps += 1

        # end task
        self.agent.end_task(id)

    def run(self):
        for id, task in enumerate(self.env.seq_tasks):
            task, hint = task['task'], task['hint']
            self.run_task(id, task, hint)


