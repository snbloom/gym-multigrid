import gym
import time
from gym.envs.registration import register
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    elif args.env == 'collect':
        register(
            id='multigrid-collect-v0',
            entry_point='gym_multigrid.envs:CollectGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-collect-v0')

    elif args.env == "attachment":
        register(
            id='multigrid-attachment-v0',
            entry_point='gym_multigrid.envs:AttachmentGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-attachment-v0')

    # attachment game can be called using args, but is also default env
    else:
        register(
            id='multigrid-attachment-v0',
            entry_point='gym_multigrid.envs:AttachmentGame4HEnv10x10N2',
        )
        env = gym.make('multigrid-attachment-v0')
        

    _ = env.reset()

    nb_agents = len(env.agents)

    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        ac = [env.action_space.sample() for _ in range(nb_agents)]

        obs, _, done, _ = env.step(ac)

        if done:
            break

if __name__ == "__main__":
    main()

# CODE FROM MINIGRID WEBSITE FOR CREATING AND RUNNING RL SYSTEMS:
# import gymnasium as gym
# env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = policy(observation)  # User-defined policy function
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()