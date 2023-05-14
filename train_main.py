from make_env_agent import make_env,make_agent
from parl.utils import logger

CRITIC_LR = 0.001
ACTOR_LR = 0.001
GAMMA = 0.95
TAU = 0.01
MAX_STEP_PER_EPISODE = 25
max_episodes = 150

def run_episode(env, agent):
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    while not done and episode_steps < MAX_STEP_PER_EPISODE:
        episode_steps += 1

        action = agent.sample(obs)
        next_obs, reward, done = env.step(action)
        terminal = float(done) if episode_steps < MAX_STEP_PER_EPISODE else 0
        agent.add_experience(obs, action, reward, next_obs, terminal)
        agent.learn()

        obs = next_obs
        episode_reward += reward
    return episode_reward, episode_steps

def train_main():
    #paddle.seed(1)
    env = make_env()
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    agent = make_agent(obs_dim,act_dim,GAMMA,TAU,ACTOR_LR,CRITIC_LR)

    total_episodes = 0
    while total_episodes <= max_episodes:
        episode_reward, episode_steps = run_episode(env, agent)
        logger.info(' episode {},reward {},episode steps {}'.format(total_episodes,episode_reward,episode_steps))
        total_episodes += 1

if __name__ == '__main__':
    train_main()

