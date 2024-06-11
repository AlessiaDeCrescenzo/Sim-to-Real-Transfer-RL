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

class Actor(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

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
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist
        
class Critic(torch.nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.state_space = state_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        value = self.fc3_critic(x_critic)

        return value


class Agent(object):
    def __init__(self, actor,critic, device='cpu',lr=1e-3):     #ho cambiato init perchè mi serve per gridsearch
        self.train_device = device
        self.actor = actor.to(self.train_device)
        self.critic = critic.to(self.train_device)
        params=[actor.parameters(), critic.parameters()] #itertools.chain(*params)
        self.optimizer = torch.optim.Adam(itertools.chain(*params), lr=lr)
        #self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=)

        self.gamma = 0.99
        self.I=1
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        I = torch.zeros_like(rewards)
        I[0]=1

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        state_values = self.critic(states)
        next_state_values = self.critic(next_states)
        next_state_values = next_state_values.squeeze(-1)
        state_values = state_values.squeeze(-1)

        for t in reversed(range(1, rewards.size(-1))):
            I[t]= I[t-1]*self.gamma

        #
        # TASK 3:
        #   compute boostrapped discounted return estimates
        returns = bootstrapped_rewards(rewards,next_state_values,self.gamma)
        #returns.detach()
        #   compute advantage terms
        with torch.no_grad():
            advantages = returns - state_values
        #   compute actor loss and critic loss
    

        

        actor_loss = -torch.mean(I * action_log_probs * advantages.detach())
        critic_loss = F.mse_loss(advantages.detach(), state_values)
        
        #   - compute gradients and step the optimizer
        # Perform optimization step
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.optimizer.step()
        
        
        #self.critic_optimizer.step()

        self.I=self.I*self.gamma

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.actor(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

