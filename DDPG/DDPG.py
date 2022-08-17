import gym
import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import gym
from gym import spaces
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import random
from gym.envs.mujoco.hopper_v4 import HopperEnv
env = HopperEnv()

#env = gym.make("Pendulum-v1")


num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):

        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

T.set_default_tensor_type('torch.DoubleTensor')

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        target_actor.eval()
        target_critic.eval()
        critic_model.eval()

        target_actions = target_actor(next_state_batch)
        y = reward_batch + gamma * target_critic(next_state_batch, target_actions)
        critic_value = critic_model(state_batch, action_batch)

        # Critic update
        critic_model.train()
        critic_model.optimizer.zero_grad()
        critic_loss = F.mse_loss(y, critic_value)
        critic_loss.backward()
        critic_model.optimizer.step()

        # Actor update
        critic_model.eval()
        actor_model.optimizer.zero_grad()
        actions = actor_model(state_batch)
        actor_model.train()
        critic_value = critic_model(state_batch, actions)

        actor_loss = -T.mean(critic_value)
        actor_loss.backward()
        actor_model.optimizer.step()

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = T.tensor(self.state_buffer[batch_indices])
        action_batch = T.tensor(self.action_buffer[batch_indices])
        reward_batch = T.tensor(self.reward_buffer[batch_indices])
        next_state_batch = T.tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

def update_target(tau=0.005):

    actor_params = actor_model.named_parameters()
    critic_params = critic_model.named_parameters()
    target_actor_params = target_actor.named_parameters()
    target_critic_params = target_critic.named_parameters()

    critic_state_dict = dict(critic_params)
    actor_state_dict = dict(actor_params)
    target_critic_dict = dict(target_critic_params)
    target_actor_dict = dict(target_actor_params)

    for name in critic_state_dict:
        critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()

    target_critic.load_state_dict(critic_state_dict)

    for name in actor_state_dict:
        actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
    target_actor.load_state_dict(actor_state_dict)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.n_actions = action_dim

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))
        outputs = mu * upper_bound

        return outputs


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dim
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.n_actions = action_dim
        self.fc1 = nn.Linear(input_dim + action_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1 = self.q1(q1_action_value)
        return q1

'''
def policy(state, noise_object):
    sampled_actions = actor_model(state).squeeze()
    noise = noise_object()

    sampled_actions = sampled_actions.detach().numpy() + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]
'''

# for mujoco
def policy(state, noise_object):
    noise = noise_object()

    sampled_actions = actor_model(state)
    sampled_actions = sampled_actions.detach().numpy() + noise
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

critic_lr = 0.002
actor_lr = 0.001

actor_model = ActorNetwork(actor_lr, num_states, num_actions)
critic_model = CriticNetwork(critic_lr, num_states, num_actions)

target_actor = ActorNetwork(actor_lr, num_states, num_actions)
target_critic = CriticNetwork(critic_lr, num_states, num_actions)

gamma = 0.99
tau = 0.005
buffer = Buffer()

ep_reward_list = []
avg_reward_list = []
total_episodes = 5000
render_freq = 100
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    #while True:
    for _ in range(500):
        #env.render()
        prev_state = T.Tensor(prev_state).unsqueeze(0)

        action = policy(prev_state, ou_noise)
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))

        episodic_reward += reward

        buffer.learn()
        update_target(tau=tau)

        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)
    avg_reward = np.mean(ep_reward_list[-10:])

    if ep % render_freq == 0:
        print("=" * 50)
        print("{}th epoch = {}".format(ep, avg_reward))
        print("=" * 50)

    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.savefig('ddpg_score_history.png')
plt.show()
plt.close()
env.close()
