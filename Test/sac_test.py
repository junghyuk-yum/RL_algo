import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import namedtuple

torch.set_default_tensor_type('torch.DoubleTensor')
torch.autograd.set_detect_anomaly(True)

Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'd'])
capacity = 10000
batch_size = 128

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        state = state.reshape(-1, self.state_dim)
        action = action.reshape(-1, self.action_dim)
        xu = torch.cat([state, action], 1)

        x = F.relu(self.fc1(xu))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, num_actions)
        self.log_std_head = nn.Linear(hidden_dim, num_actions)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head

class SAC(object):
    def __init__(self, state_dim, action_dim, capacity = capacity, lr = 3e-4):
        super(SAC, self).__init__()

        self.policy_net = Actor(state_dim, action_dim)
        self.value_net = Critic(state_dim)
        self.Target_value_net = Critic(state_dim)
        self.Q_net1 = Q(state_dim, action_dim)
        self.Q_net2 = Q(state_dim, action_dim)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.Q1_optimizer = optim.Adam(self.Q_net1.parameters(), lr=lr)
        self.Q2_optimizer = optim.Adam(self.Q_net2.parameters(), lr=lr)

        self.replay_buffer = [Transition] * capacity
        self.num_transition = 0 # pointer of replay buffer
        self.num_training = 1

        self.value_criterion = nn.MSELoss()
        self.Q1_criterion = nn.MSELoss()
        self.Q2_criterion = nn.MSELoss()

        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self._scale_reward = 3


    def select_action(self, state):
        state = torch.Tensor(state)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.sample()
        action = torch.tanh(z).detach().numpy()
        return action

    def store(self, s, a, r, s_, d):
        index = self.num_transition % capacity
        transition = Transition(s, a, r, s_, d)
        self.replay_buffer[index] = transition
        self.num_transition += 1

    def evaluate(self, state):
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(batch_mu + batch_sigma*z)
        log_prob = dist.log_prob(batch_mu + batch_sigma * z) - torch.log(1 - action.pow(2) + torch.tensor(1e-7))

        return action, log_prob, z, batch_mu, batch_log_sigma

    def update(self):

        s = torch.tensor([t.s for t in self.replay_buffer])
        a = torch.tensor([t.a for t in self.replay_buffer])
        r = torch.tensor([t.r for t in self.replay_buffer])
        s_ = torch.tensor([t.s_ for t in self.replay_buffer])
        d = torch.tensor([t.d for t in self.replay_buffer])

        index = np.random.choice(range(capacity), batch_size, replace=False)
        bn_s = s[index]
        bn_a = a[index]
        bn_r = r[index].reshape(-1, 1)
        bn_s_ = s_[index]
        bn_d = d[index].int().reshape(-1, 1)

        alpha, gamma = 0.2, 0.99
        target_value = self.Target_value_net(bn_s_) # V_bar{psi}(st+1)
        next_q_value = bn_r * self._scale_reward + (1 - bn_d) * gamma * target_value # Qˆ(st, at )

        # Get targets
        excepted_value = self.value_net(bn_s) # V(s_t)
        excepted_Q1, excepted_Q2 = self.Q_net1(bn_s, bn_a), self.Q_net2(bn_s, bn_a) #JQ (θi )  for i  ∈  {1, 2}
        sample_action, log_prob, z, batch_mu, batch_log_sigma = self.evaluate(bn_s)

        policy_Q1, policy_Q2 = self.Q_net1(bn_s, sample_action), self.Q_net2(bn_s, sample_action)
        excepted_min_Q = torch.min(policy_Q1, policy_Q1) # Qθ (st , at )
        next_value = excepted_min_Q - log_prob # Eat ∼πφ [Qθ (st , at ) − log πφ (at |st )]

        def value_update():
            V_loss = self.value_criterion(excepted_value, next_value.detach()).mean()  # J_V

            # mini batch gradient descent
            self.value_optimizer.zero_grad()
            V_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

        value_update()

        def Q_value_update():
            # Dual Q net
            Q1_loss = self.Q1_criterion(excepted_Q1, next_q_value.detach()).mean()  # J_Q
            Q2_loss = self.Q2_criterion(excepted_Q2, next_q_value.detach()).mean()  # 0.5 Qθ(st, at) − Qˆ (st, at)2

            self.Q1_optimizer.zero_grad()
            Q1_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net1.parameters(), 0.5)
            self.Q1_optimizer.step()

            self.Q2_optimizer.zero_grad()
            Q2_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.Q_net2.parameters(), 0.5)
            self.Q2_optimizer.step()

        Q_value_update()

        def policy_update():
            # Pi update
            mu, log_sigma = self.policy_net(bn_s)
            sigma = torch.exp(log_sigma)
            noise = Normal(0, 1)
            z = noise.sample()
            action = torch.tanh(mu + sigma * z)

            actor_Q1, actor_Q2 = self.Q_net1(bn_s, action), self.Q_net2(bn_s, action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            pi_loss = (log_prob - actor_Q).mean()  # Jπ (φ) = Est ∼D,t ∼N [log πφ (fφ (t ; st )|st ) − Qθ (st , fφ (t ; st ))]

            self.policy_optimizer.zero_grad()
            pi_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

        policy_update()

        # update target v net update
        tau = 0.005
        for target_param, param in zip(self.Target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param * (1 - tau) + param * tau)

        self.num_training += 1