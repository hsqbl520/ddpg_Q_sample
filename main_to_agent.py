import parl
import paddle
from parl.utils import ReplayMemory

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim_n,act_dim_n):
        super(Agent, self).__init__(algorithm)
        self.algorithm = algorithm
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = 128

        self.memory_size = int(1e5)
        self.min_memory_size = self.batch_size * 25
        self.rpm = ReplayMemory(
            max_size=self.memory_size,obs_dim=self.obs_dim_n,act_dim=self.act_dim_n)
        self.global_train_step = 0


    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        act = self.algorithm.predict(obs)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def sample(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        act = self.algorithm.sample(obs)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self):
        self.global_train_step += 1
        #if self.global_train_step % 100 != 0:
            #return 0.0
        if self.rpm.size() <= self.min_memory_size:
            return 0.0
        rpm_sample_index = self.rpm.make_index(self.batch_size)
        batch_obs, batch_act, batch_rew, batch_obs_next, batch_isOver = self.rpm.sample_batch_by_index(rpm_sample_index)
        batch_obs = paddle.to_tensor(batch_obs, dtype='float32')
        batch_act = paddle.to_tensor(batch_act, dtype='float32')
        batch_rew = paddle.to_tensor(batch_rew, dtype='float32')
        batch_isOver = paddle.to_tensor(batch_isOver, dtype='float32')
        batch_obs_next = paddle.to_tensor(batch_obs_next, dtype='float32')

        self.algorithm.learn(batch_obs, batch_act, batch_rew,batch_obs_next,batch_isOver)

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)