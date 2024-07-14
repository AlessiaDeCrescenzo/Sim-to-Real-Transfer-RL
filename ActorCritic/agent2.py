import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import itertools

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def bootstrapped_rewards(r,v,gamma):
    bootstrapped_r= torch.zeros_like(r)
    for t in reversed(range(0, r.size(-1))):
        bootstrapped_r[t]= r[t] + gamma*v[t]
    return bootstrapped_r

class ActorCritic(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden=64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc4_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.6
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc4_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        #x_actor = self.tanh(self.fc3_actor(x_actor))
        action_mean = self.fc4_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc4_critic(x_critic)

        return normal_dist,value

class Agent(object):
    def __init__(self, actorcritic, device='cpu', lr=1e-3, gamma=0.99, ewc_lambda=0.4):
        self.train_device = device
        self.actorcritic = actorcritic.to(self.train_device)
        self.optimizer = torch.optim.Adam(actorcritic.parameters(), lr=lr)
        self.gamma = gamma
        self.ewc_lambda = ewc_lambda

        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        self.prev_params = {name: param.clone() for name, param in actorcritic.named_parameters()}
        self.fisher = {name: torch.zeros_like(param) for name, param in actorcritic.named_parameters()}

    def compute_fisher(self, data_loader):
        self.actorcritic.eval()
        for inputs, _ in data_loader:
            self.actorcritic.zero_grad()
            inputs = inputs.to(self.train_device)
            normal_dist, values = self.actorcritic(inputs)
            loss = normal_dist.log_prob(inputs).sum()  # Assuming inputs are actions taken
            loss.backward()
            for name, param in self.actorcritic.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
        for name in self.fisher:
            self.fisher[name] /= len(data_loader)

    def ewc_loss(self):
        loss = 0
        for name, param in self.actorcritic.named_parameters():
            if param.requires_grad:
                loss += (self.fisher[name] * (param - self.prev_params[name]) ** 2).sum()
        return self.ewc_lambda / 2 * loss

    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        _, state_values = self.actorcritic(states)
        _, next_state_values = self.actorcritic(next_states)
        next_state_values = next_state_values.squeeze(-1)
        state_values = state_values.squeeze(-1)

        returns = bootstrapped_rewards(rewards, next_state_values, self.gamma)
        with torch.no_grad():
            advantages = returns - state_values

        actor_loss = -torch.mean(action_log_probs * advantages.detach())
        critic_loss = F.mse_loss(advantages, state_values)
        loss = actor_loss + critic_loss + self.ewc_loss()

        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actorcritic.parameters(), 1)
        self.optimizer.step()

        # Update prev_params after the optimizer step
        self.prev_params = {name: param.clone() for name, param in self.actorcritic.named_parameters()}

        return

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.actorcritic(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None
        else:  # Sample from the distribution
            action = normal_dist.sample()
            action_log_prob = normal_dist.log_prob(action).sum()
            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)