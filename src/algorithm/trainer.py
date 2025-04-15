class TaskTrainer:
    def __init__(self, agent, environment, config):
        # self.config = config.algorithm
        self.agent = agent
        self.env = environment

    def run_task(self, id, task, hint):
        env = self.env.get_single_env(task)

        obs, info = env.reset()  # Reset environment
        act = env.action_space.sample()  # Sample an action
        for i in range(10):
            obs, reward, terminate, truncate, info = env.step(act)
        print(f'{id}. {task}: {hint}')
        print(f'obs: {obs.shape}, act: {act.shape}')


    def run(self):
        for id, task in enumerate(self.env.seq_tasks):
            task, hint = task['task'], task['hint']
            self.run_task(id, task, hint)


