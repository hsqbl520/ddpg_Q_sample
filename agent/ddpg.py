import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from copy import deepcopy

class DDPG(parl.Algorithm):
    def __init__(self,model,
                 gamma=None,tau=None,
                 actor_lr=None,critic_lr=None):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)
        self.sync_target(0)

        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=self.actor_lr,parameters=self.model.get_actor_params(),grad_clip=nn.ClipGradByNorm(clip_norm=0.5))
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=self.critic_lr,parameters=self.model.get_critic_params(),grad_clip=nn.ClipGradByNorm(clip_norm=0.5))

    def predict(self, obs):
        policy = self.model.policy(obs)
        mean = policy[0]
        action = paddle.tanh(mean)
        return action

    def sample(self, obs):
        policy = self.model.policy(obs)

        mean, std = policy[0], paddle.exp(policy[1])
        mean_shape = paddle.to_tensor(mean.shape, dtype='int64')
        random_normal = paddle.normal(shape=mean_shape)
        action = mean + std * random_normal
        action = paddle.tanh(action)
        return action

    def learn(self, obs, act, reward, obs_next, terminal):
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, act, reward, obs_next, terminal)
        self.sync_target()
        return critic_cost,actor_cost

    def _actor_learn(self, obs):

        action_input = self.sample(obs)
        eval_q = self.model.value(obs, action_input)
        cost = paddle.mean(-1.0 * eval_q)

        self.actor_optimizer.clear_grad()
        cost.backward()
        self.actor_optimizer.step()
        return cost

    def _critic_learn(self, obs, act, reward, obs_next, terminal):
        target_act_next = self.sample(obs_next)
        target_act_next = target_act_next.detach()
        target_q_next = self.target_model.value(obs_next, target_act_next)
        target_q = reward + self.gamma * (1.0 - terminal) * target_q_next.detach()

        pred_q = self.model.value(obs, act)
        cost = paddle.mean(F.square_error_cost(pred_q, target_q))

        self.critic_optimizer.clear_grad()
        cost.backward()
        self.critic_optimizer.step()
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
