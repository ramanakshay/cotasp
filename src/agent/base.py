
class TaskAgent:
    '''
    Batch Update Agent
    '''
    def start_task(self, id, hint):
        '''
        Notify agent of start of new task
        '''
        raise NotImplementedError('Not implemented `start_task` method for agent.')

    def end_task(self, id):
        '''
            Notify agent of end of new task
        '''
        raise NotImplementedError('Not implemented `end_task` method for agent.')

    def update(self, batch, id=None):
        '''
            Update agent from a batch of experiences
        '''
        raise NotImplementedError('Not implemented `update` method for agent.')

    def sample_actions(self, obs, id=None):
        '''
            Given task-id and current observation, predict action
        '''
        raise NotImplementedError('Not implemented `sample_action` method for agent.')
