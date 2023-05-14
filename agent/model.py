import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Model(parl.Model):
    def __init__(self,obs_dim,act_dim):
        super(Model, self).__init__()
        self.actor_model = ActorModel(obs_dim, act_dim)
        self.critic_model = CriticModel(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ActorModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        std_hid_size = 64
        self.fc1 = nn.Linear(obs_dim,hid1_size)
        self.fc2 = nn.Linear(hid1_size,hid2_size)
        self.fc3 = nn.Linear(hid2_size,act_dim)
        self.std_fc = nn.Linear(std_hid_size,act_dim)

    def forward(self, obs):
        hid1 = F.relu(self.fc1(obs))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        act_std = self.std_fc(hid2)
        return (means, act_std)

class CriticModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(CriticModel, self).__init__()
        hid1_size = 64
        hid2_size = 64
        out_dim = 1
        self.fc1 = nn.Linear(obs_dim+ act_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, out_dim)

    def forward(self, obs_n, act_n):
        inputs = paddle.concat([obs_n , act_n], axis=1)
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        Q = paddle.squeeze(Q, axis=1)
        return Q
