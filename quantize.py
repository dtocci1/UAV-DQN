'''
This will trim our network and reduce weights
May encounter errors as we are using RL instead of CNN

https://github.com/Xilinx/Vitis-AI-Tutorials/blob/master/Design_Tutorials/09-mnist_pyt/files/quantize.py
'''

import os
import sys
import argparse
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
    
from common import *

def quantize(build_dir,quant_mode,batchsize):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    device = torch.device('cpu') # may replace with GPU if we ever get it working

    # load trained model
    model = neural_network().to(device)
    model.load_state_dict(torch.load(os.path.join(float_model,'f_model.pth')))

    # force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # override batchsize if in test mode
    if (quant_mode=='test'):
        batchsize = 1
    
    rand_in = torch.randn([batchsize, 1, 28, 28])
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model


    # data loader
    test_dataset = torchvision.datasets.MNIST(dset_dir,
                                                train=False, 
                                                download=True,
                                                transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batchsize, 
                                                shuffle=False)

    # evaluate 
    test(quantized_model, device, test_loader)


    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
    
    return


def run_main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
    ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b',  '--batchsize',  type=int, default=100,        help='Testing batchsize - must be an integer. Default is 100')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--quant_mode   : ',args.quant_mode)
    print ('--batchsize    : ',args.batchsize)
    print(DIVIDER)

    quantize(args.build_dir,args.quant_mode,args.batchsize)

    return



if __name__ == '__main__':
    run_main()