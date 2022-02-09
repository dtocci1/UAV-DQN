'''
UAV ENV:

This environment is based on a UAV inspecting packets continually thoughout the day. Thus, its decisions are
based on what it has seen previously in the day from sources, and what actions have led to what rewards (action-space).
The action will either allow the packet as benign, or reject it and identify

Observation:
    Packet data from training set (load in one column)

Action:
        allow packet
            0: allow

        OR identify
            1: portscan
            2: ddos
            3: bot
            4: infiltration
            5: bruteforce
            6: xss
            7: sql
            8: ftp
            9: ssh
            10: dos
            11: heartbleed

Reward:
    Server will return data relative to the packet
        This will mimic the server going down from a malicious packet (which the UAV could in theory detect),
        or this will mimic and end user reporting to the network he could not connect through
        Allows for quicker setup, a more in depth env may be needed

Step:
    Get reward for action
    load next state or return DONE
    next state comes from 1-3 excel data

Reset:
    Clear all variables
    Reset run counter
    Load in one packet

Loading in data:
    We have 3 CSV file as one was too big for openOffice, can prob merge manually
    Should load in one data point at a time
    Main function will generate a history of a certain length
        D = [s,a,r,s+1],[] ... x N, need to determine N value... 1000?
    Main will then sample one state action space from D, calculate loss and back propogate


Sources:
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    https://github.com/apoddar573/Tic-Tac-Toe-Gym_Environment/blob/master/gym-tictac4/gym_tictac4/envs/tictac4_env.py

TO DO:
    - Finish adjusting batch size capabilities
    - Should env end on fixed time intervals?
    - Alternatively, could have it converge if a certain number of bad packets go through
        - heavily penalize?
'''
import gym
import csv
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class UavEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.cur_packet = 1 # Tracker for what part of the day we're on
        self.action_reward = {0: "benign",
                    1: "portscan",
                    2: "ddos",
                    3: "bot",
                    4: "infiltration",
                    5: "bruteforce",
                    6: "xss",
                    7: "sql",
                    8: "ftp",
                    9: "ssh",
                    10: "dos",
                    11: "heartbleed"}
        self.action_space = spaces.Discrete(12)
        ''' Action space choices:
            allow packet
            0: allow

            OR identify
            1: portscan
            2: ddos
            3: bot
            4: infiltration
            5: bruteforce
            6: xss
            7: sql
            8: ftp
            9: ssh
            10: dos
            11: heartbleed

        '''
        self.observation_space = spaces.Box(np.array([0,0,0,0,0,0,0,0,0]), np.array([65535,10000000,10000,10000,10000,10000,10000,10000,10000])) # observation data, num depends on input paramers
        self.reward = 0
        self.done = False
        self.runs = 0
        self.batch_size = 0

        with open("dataset/total.csv","r") as file:
            self.data = list(csv.reader(file))
        self.classifier = self.data[self.cur_packet][9]

    def step(self, action):

        if (action == 0) and (self.classifier == "benign"): # Allows benign packet through
            self.reward = 1
        elif (action == 0) and (self.classifier != "benign"): # Allows malicious packet through
            self.reward = -10 # Heavy penalty for false negative
        elif self.action_reward[action] == self.classifier: # Properly identified attack
            self.reward = 5
        else:
            self.reward = -5 # Misidentified attack, but still didn't allow it through

        self.cur_packet = self.cur_packet + 1

        # Check if we've finished batch, or if somehow reached end of dataset

        if ((self.cur_packet % batch_size) != 0) and (self.cur_packet <= len(self.data) - 1):
            self.observation_space = [float(i) for i in self.data[self.cur_packet][:8]]
            self.classifier = self.data[self.cur_packet][9]
        else:
            self.done = True

        return self.observation_space, self.reward, self.done

    def reset(self, batchsize):
        self.observation_space = spaces.Box(np.array([0,0,0,0,0,0,0,0,0]), np.array([65535,10000000,10000,10000,10000,10000,10000,10000,10000])) # observation data, num depends on input paramers
        self.action = 0
        self.reward = 0
        #self.runs += 1 # Track run for batching
        self.done = False
        self.observation_space = [float(i) for i in self.data[self.cur_packet][:8]]
        self.classifier = self.data[self.cur_packet][9]

        return self.observation_space

    def render():
        pass
