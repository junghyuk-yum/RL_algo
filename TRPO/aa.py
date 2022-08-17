import torch
import torch.nn as nn
import gym
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dim
        self.n_actions = action_dim

        self.fc1_dims = 256
        self.fc2_dims = 256
        self.fc1 = nn.Linear(input_dim + action_dim, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = self.fc1(torch.cat([state, action], dim=1 ))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        val = self.q1(x)
        return val


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, action_dim):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = action_dim
        self.fc1_dims = 256
        self.fc2_dims = 256

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mu = torch.sigmoid(self.mu(x))
        return mu

class ReplayBuffer():

    def __init__(self, max_size, input_dims, action_dim):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_dims))
        self.next_state_memory = np.zeros((self.mem_size, input_dims))
        self.action_memory = np.zeros((self.mem_size, action_dim))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_len = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_len, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones


class Agent(object):
    def __init__(self, alpha, beta, tau, env, input_dims, n_actions,
                 gamma=0.99, max_size=1000000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.actor = ActorNetwork(alpha, input_dims, n_actions)
        self.critic = CriticNetwork(beta, input_dims, n_actions)
        self.target_actor = ActorNetwork(alpha, input_dims, n_actions)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions)

        self.scale = 1.0
        self.noise = np.random.normal(scale=self.scale, size=(n_actions))
        self.update_network_parameters(tau=tau)

    def choose_action(self, observation):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise, dtype=torch.float)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done).to(self.critic.device)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state).detach()
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * (not done[j]))
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1).detach()

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = Agent(alpha=0.025, beta=0.025, tau=0.005, env=env,
              batch_size=100, input_dims=state_dim, n_actions=action_dim)

score_history = []
epochs = 1500
render_frequency = 100
for i in range(epochs):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)

    if i % render_frequency == 0:
        print("=" * 50)
        print("{}th epoch = {}".format(i , score))
        print("=" * 50)


import matplotlib.pyplot as plt
plt.plot(range(1, len(score_history)+1), score_history)
plt.savefig('ddpg_score_history.png')
plt.close()
