from algorithm.replay_buffer import ReplayBuffer
from algorithm.evaluator import TaskEvaluator

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
        max_steps = self.config.max_steps
        is_task_aware = self.config.is_task_aware
        print(f'Learning on task {id}: {task} for {max_steps} steps')
        env = self.env.get_single_env(task)
        self.replay_buffer.reset()
        # start task
        if is_task_aware:
            self.agent.start_task(id, hint)
        else:
            id = None

        obs, info = env.reset()  # Reset environment
        for i in range(max_steps):
            # sample action
            if (i < self.config.start_training):
                # initial exploration strategy proposed in ClonEX-SAC
                if id == 0:
                    action = self.env.action_space.sample()
                else:
                    # choose random previous task (uniform prev strategy)
                    rand_id = np.random.choice(id) if is_task_aware else None
                    action = agent.sample_action(obs, rand_id)
                    action = np.asarray(action, dtype=np.float32).flatten()
            else:
                action = agent.sample_actions(obs, id)
                action = np.asarray(action, dtype=np.float32).flatten()

            # take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # add to buffer
            self.replay_buffer.insert(obs, action, reward, float(done), next_obs)

            # repeat
            obs = next_obs
            if done:
                obs, info = env.reset()

            # train
            if (i >= self.config.start_training) and (i % self.config.train_interval == 0):
                for _ in range(self.config.train_updates):
                    batch = self.replay_buffer.sample()
                    update_info = self.agent.update(batch, id)

            #eval
            if i % self.config.eval_interval == 0:
                stats = self.evaluator.run()
                print(stats)

            # increment global step
            self.total_env_steps += 1

        # end task
        if is_task_aware:
            self.agent.end_task(id)

    def run(self):
        for id, task in enumerate(self.env.seq_tasks):
            task, hint = task['task'], task['hint']
            self.run_task(id, task, hint)


