from gym import spaces
import numpy as np

class AgentEnv():
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None,done_callback=None):

        self.world = world
        self.agent = self.world.agent
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.done_callback = done_callback

        self.action_space = spaces.Box(low=-self.agent.amax, high=+self.agent.amax, shape=(world.dim_p,), dtype=np.float32)
        obs_dim = len(observation_callback(self.agent, self.world))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)##obs_dim行的box

    def step(self, action):
        self.agent.action.u = action
        self.world.step()

        obs = self.observation_callback(self.agent, self.world)
        reward = self.reward_callback(self.agent, self.world)
        done = self.done_callback(self.agent, self.world)
        return obs, reward, done

    def reset(self):
        self.reset_callback(self.world)
        return self.observation_callback(self.agent, self.world)
