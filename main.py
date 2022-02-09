'''
Vitis-AI names this file
    "common.py", may need to follow suit.

Sets up neural network, train, and test functions

Source code: https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0


USING Gymp version 0.15.4, may cause significant incompatibilities

TO DO:
    - Readjust train function
    - screw code from towardsdatascience it sucks fucking dick butt
    - follow DQN paper strat, we'll minibatch and sample
    - should we have a batch size out of csv, and then minibatch for experience replay?
        - or do we just continually retrain over ENTIRE data set, with mini batching
    - do we or should we randomize data?
'''

import numpy as np
#import matplotlib.pyplot as plt
import sys
import torch
import gym
import uavenv

from torch import nn
from torch import optim

class neural_network():
    def __init__(self):
        super(neural_network,self).__init__() # May be needed for vitis-ai
        self.n_inputs = 8
        self.n_outputs = 12

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 32),
            nn.LayerNorm(32), # need to normalize inputs, as they are huge
            nn.ReLU(),
            nn.Linear(32,self.n_outputs),
            nn.Softmax()
        )

    def forward(self, state): # Naming for vitis-ai
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::1].cumsum()[::-1]
    return r


def train(model, env, optimizer_pass, epochs,batch_size,gamma=0.99): 
    '''
        Train should work as follows for typical DQN according to paper:
            load state, and do any preprocessing (GYM function)
            with some greedy probability (epsilon), select an action
            take the action and observe reward as well as new state (GYM function)
            Append to D <= [p_state_t, action, reward, p_state_t+1]
            Sample Xi minibatch from D
                Calculate Yj for Xi
                    Yj = reward at state t + Q*(t+1), aka our known reward plus predicted future
                LOSS = Yj - Q*(t), what we thought reward would be from t
                    we know this as t has already been exectued, reward has been received
                perform gradient descent (use optimzer)
    '''
    # Results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    #model.train() # May be recursive, done in vitis-ai example
    optimizer = optimizer_pass

    action_space = np.arange(env.action_space.n)
    ep = 0 # episodes/epochs

    while ep < epochs: 
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False

        while done == False:
            # Get actions and convert to numpy array
            action_probs = model.forward(s_0).detach().numpy() # predicted Q-value of [0 or 1] => [Rallow, Rblock]
            action = np.random.choice(action_space, p=action_probs) # action is weighted random based on expected reward
            s_1, r, done = env.step(action, batch_size) # Step and get reward for this
            
            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1
            
            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
               # print(batch_rewards[-1])
                batch_states.extend(states)
                batch_actions.extend(actions)
                gathered_actions = [[action] for action in batch_actions] # neat way to make torch.gather work
                    
                batch_counter += 1
                total_rewards.append(sum(rewards))
                
                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(batch_rewards)
                    #print(reward_tensor)
                    # Actions are used as indices, must be 
                    # LongTensor
                    action_tensor = torch.LongTensor(gathered_actions)

                    #state_action_values = model.network(batch_states)

                    # Calculate loss
                    states_chosen = model.network.forward(state_tensor) # UAV env decisions being 0 or -inf fucks this up bad
                    #print(states_chosen)
                    selected_probs = reward_tensor * torch.gather(states_chosen, 1, action_tensor).squeeze() # State actions selected
                    #print(selected_probs)
                    loss = selected_probs.mean()
                    print("LOSS: ", loss)
                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()
                    
                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1
                
                avg_rewards = np.mean(total_rewards[-100:])
                #print(total_rewards[-100:])
                #Print running average
                print("\rEp: {} Average of last 100:" +   
                     "{:.2f}".format(
                     ep + 1, avg_rewards), end="")
                ep += 1

def test():
    NULL


print("Program works still.") # Sanity check for pyTorch1.4.0 compatibility