from main_to_env import AgentEnv
from env.simple_spread import Scenario

from main_to_agent import Agent
from agent.ddpg import DDPG
from agent.model import Model
def make_env():

    scenario = Scenario()
    world = scenario.make_world()
    scenario.reset_world(world)
    env = AgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,scenario.done)

    return env

def make_agent(obs_dim, act_dim,GAMMA,TAU,ACTOR_LR,CRITIC_LR):
    model = Model(obs_dim, act_dim)
    algorithm = DDPG(
        model,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = Agent(
        algorithm,
        obs_dim_n=obs_dim,
        act_dim_n=act_dim)

    return agent
