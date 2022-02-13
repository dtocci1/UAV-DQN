import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys
import os
import shutil
import gym
import yaml

from main import *



def train_test(build_dir, batchsize, gamma, epochs):
    dset_directory = build_dir + '/dataset'
    float_model = build_dir + '/float_model' # float weights, not yet INT8

    #device = torch.device('cpu')

    #model = neural_network().to(device)

    # Load in datasets, set up "env"
    env = gym.make("uavenv-v0")
    neural_net = neural_network()

    optimizer = optim.Adam(neural_net.network.parameters(), lr=gamma)

    # Train the model
    train(neural_net,env,optimizer,epochs,batchsize,gamma)

    # Save the trained model
    shutil.rmtree(float_model,ignore_errors=True)
    os.makedirs(float_model)
    save_path = os.path.join(float_model,'f_model.pth')
    torch.save(neural_net.network.state_dict(),save_path)
    print('Trained model saved to ', save_path)

    return


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=1000,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=10000,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    args = ap.parse_args()

    train_test(args.build_dir, args.batchsize, args.learnrate, args.epochs)

    return

if __name__  == '__main__':
    main()