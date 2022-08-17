from collections import namedtuple
from torch.distributions import Categorical
from torch.optim import Adam
from random import random
from bisect import bisect
import numpy as np

import torch.nn.functional as F
import gym
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

state_size = env.observation_space.shape[0]
num_actions = env.action_space.n

Rollout = namedtuple('Rollout',
                     ['states', 'actions', 'rewards', 'next_states'])


def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0)).detach()
    next_values = torch.zeros_like(rewards).detach()

    # back-propagate, gamma = 0.99
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
        #print(last_value)
    advantages = next_values - values
    return advantages

def GAE(states, last_state, rewards):
    gamma = 0.99
    lamda = 0.95

    values = critic.forward(states)
    last_value = critic(last_state.unsqueeze(0)).detach()
    next_values = torch.zeros_like(rewards).detach()
    advantages = torch.zeros(rewards.size()[0]+1).unsqueeze(1)

    # back-propagate, gamma = 0.99
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
        delta = rewards[i] + next_values[i] - values[i]
        advantages[i] = delta + (gamma * lamda * advantages[i + 1])

    return advantages[:rewards.size()[0]]

def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()

def kl_div(p, q):
    return (p * (p.log() - q.log())).sum(-1).mean()

def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def hessian_vector_product(d_kl, value, parameters):
    return flat_grad(d_kl @ value, parameters, retain_graph=True)

def conjugate_gradient(hessian_vector_product, g, d_kl, parameters, delta=0., max_iterations=10):
    # hessian_vector_product = A, g = b
    x = torch.zeros_like(g) # x_0
    # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    r = g.clone()
    p = r.clone()
    r_dot_old = r @ r

    for _ in range(max_iterations) :
        z = hessian_vector_product(d_kl, p, parameters)
        alpha = r_dot_old / (p @ z)
        x_new = x + alpha * p
        r = r - alpha * z
        beta = (r @ r) / r_dot_old
        p = r + beta * p

        x = x_new

    return x

def apply_update(step):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = step[n:n + numel].view(p.shape)
        p.data += g
        n += numel

delta = 0.001

def update_agent(rollouts):
    '''
    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, actions, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()
    '''

    advantages = [GAE(states, next_states[-1], rewards) for states, actions, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    update_critic(advantages)

    states = torch.cat([r.states for r in rollouts], dim=0) # s ~ rho_old
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten() # a ~ q

    distribution = actor.forward(states)
    if (torch.isnan(distribution).any()) : return False
    distribution = torch.distributions.utils.clamp_probs(distribution)
    print(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]
    print(probabilities)

    # Calculate the gradient
    L = surrogate_loss(probabilities, probabilities.detach(), advantages) # pi / q * advantages
    KL = kl_div(distribution.detach(), distribution)
    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True) # A_ij = delta/delta_theta_i * delta/delta_theta_j D_KL(theta_old,theta)
    try:
        x = conjugate_gradient(hessian_vector_product, g, d_kl, parameters)
        hessian = hessian_vector_product(d_kl, x, parameters)
    except:
        return False

    alpha = torch.sqrt(2 * delta / (x @ hessian))
    max_step = alpha * x

    def set_and_eval(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor.forward(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution.detach(), distribution_new)

        L_improvement = L_new - L

        # constaint
        if L_improvement > 0 and KL_new <= delta:
            return True

        apply_update(-step)
        return False

    backtrack_iters = 0
    backtrack_coeff = 0.99
    while not set_and_eval((backtrack_coeff ** backtrack_iters) * max_step) and backtrack_iters < 10:
        backtrack_iters += 1

def get_action(state):

    state = torch.tensor(state).unsqueeze(0)
    dist = actor.forward(state).squeeze(0).numpy()
    dist = np.nan_to_num(dist)

    # random choice with distribution
    total = 0
    cum_probs = []
    for p in dist:
        total += p
        cum_probs.append(total)

    if cum_probs[-1] != 1.0:
        cum_probs[-1] = 1.0

    x = random() * total - 0.0001

    i = bisect(cum_probs, x)
    return i

def train(epochs=1, num_rollouts=1, render_frequency=None):
    rollout_num = 0
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        rollouts = []

        for t in range(num_rollouts):
            state = env.reset()
            done = False

            samples = []

            while not done:

                if render_frequency is not None and render_frequency != 0 and rollout_num % render_frequency == 0:
                    env.render()

                with torch.no_grad():
                    action = get_action(state)
                    next_state, reward, done, _ = env.step(action)
                    # Collect samples
                    samples.append((state, action, reward, next_state))

                    state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)
            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)

            rollouts.append(Rollout(states, actions, rewards, next_states))


            rollout_total_rewards.append(rewards.sum().item())
            rollout_num += 1
        rollout_mean = sum(rollout_total_rewards[epoch:epoch + num_rollouts]) / len(rollout_total_rewards[epoch:epoch + num_rollouts])
        mean_total_rewards.append(rollout_mean)
        update_agent(rollouts)

class Actor(nn.Module):
    def __init__(self, state_size, actor_hidden, num_actions):
        super(Actor, self).__init__()
        self.actor_hidden1 = actor_hidden
        self.actor_hidden2 = 64
        self.num_actions = num_actions
        self.state_size = state_size

        self.l1 = nn.Linear(state_size, self.actor_hidden1)
        self.l2 = nn.Linear(self.actor_hidden1, self.actor_hidden2)
        self.l3 = nn.Linear(self.actor_hidden2, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = self.softmax(x)
        return x

actor_hidden = 128
actor = Actor(state_size, actor_hidden, num_actions)


class Critic(nn.Module):
    def __init__(self, state_size, critic_hidden , num_actions):
        super(Critic, self).__init__()
        self.critic_hidden1 = critic_hidden
        self.critic_hidden2 = 64
        self.state_size = state_size

        self.l1 = nn.Linear(self.state_size , self.critic_hidden1)
        self.l2 = nn.Linear(self.critic_hidden1, self.critic_hidden2)
        self.l3 = nn.Linear(self.critic_hidden2, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

# Critic
critic_hidden = 128
critic = Critic(state_size, critic_hidden, num_actions)
critic_optimizer = Adam(critic.parameters(), lr=0.01)

rollout_total_rewards = []
mean_total_rewards = []
train(epochs=1, num_rollouts=1, render_frequency=0)

plt.plot(range(1, len(mean_total_rewards)+1), mean_total_rewards)
plt.savefig('reward.png')
plt.close()