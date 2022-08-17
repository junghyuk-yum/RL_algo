import matplotlib.pyplot as plt
import torch
import numpy as np
from gym.envs.mujoco.humanoid_v4 import HumanoidEnv
torch.set_default_tensor_type('torch.DoubleTensor')


seed = 123456
env = HumanoidEnv()

#env = gym.make("Pendulum-v1")
env.action_space.seed(seed)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 256

from sac_test import SAC
agent = SAC(state_dim = env.observation_space.shape[0], action_dim= env.action_space.shape[0])

# Training Loop
total_numsteps = 0
updates = 0

ep_reward_list = []
avg_reward_list = []
total_avg_reward_list = []
total_episodes = 200000
render_freq = 1000

for ep in range(total_episodes):

    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset(seed=seed)

    for _ in range(1000):
        if 10000 > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        next_state, reward, done, _ = env.step(action) # Step
        agent.store(state, action, reward, next_state, done)

        episode_reward += reward

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        mask = 1
        if done:
            mask = 0

        state = next_state

        if done: break

    # buffer capacity
    if agent.num_transition >= 10000:
        agent.update()

    ep_reward_list.append(episode_reward)
    avg_reward = np.mean(ep_reward_list[-50:])

    if ep % render_freq == 0:
        print("=" * 50)
        print("{}th epoch = {}".format(ep, avg_reward))
        print("=" * 50)

    total_avg_reward_list.append(np.mean(ep_reward_list[-30:]))

plt.plot(ep_reward_list, alpha = 0.4)
plt.plot(total_avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.savefig('sac_test_score_history.png')
plt.show()
env.close()