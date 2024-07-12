import gym

class TrackRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TrackRewardWrapper, self).__init__(env)
        self.buffer = []  # keep track of experienced returns
        self.succ_metric_buffer = []  # buffer with metric used for measuring success
        self.exposed_cum_reward = 0
        self.ready_to_update_buffer = False
        self.expose_episode_stats = True

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.exposed_cum_reward += reward
        
        if done:
            if self.expose_episode_stats:
                self.buffer.append(self.exposed_cum_reward)
                self.succ_metric_buffer.append(self.exposed_cum_reward)  # Assuming the metric is the cumulative reward
                self.ready_to_update_buffer = False

        return observation, reward, done, info

    def reset(self, **kwargs):
        self.exposed_cum_reward = 0
        self.ready_to_update_buffer = True
        return self.env.reset(**kwargs)