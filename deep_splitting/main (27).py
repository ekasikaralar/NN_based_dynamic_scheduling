import json
import munch
import os
import argparse

import torch 
import numpy as np
from torch import nn
import pandas as pd

import equation as eqn
from solver import DSPL_solver
from Net import Y_Net
from Net import Z_Net

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type = str, help = "The path to load json file")
    parser.add_argument('--run_name', type = str, help = 'The name of numerical experiments')
    
    args = parser.parse_args()

    return args


def load_config(args):

    with open(args.config_path) as json_data_file:
        config = json.load(json_data_file)

    return munch.munchify(config)

def prepare_directories(args):

    os.makedirs(f"logs_{args.run_name}", exist_ok = True)
    os.makedirs(f"cppweights_{args.run_name}", exist_ok = True)


def main():
    
    print(torch.__version__)
    args = parse_arguments()
    
    config = load_config(args)
    prepare_directories(args)

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    
    dpsl_solver = DSPL_solver(config, bsde)
    
    final_iterations = config.net_config.final_iterations
    inner_iterations = config.net_config.inner_iterations

    save_path_logs = f"logs_{args.run_name}/"
    save_path_cppweights = f"cppweights_{args.run_name}/"

    training_history = dpsl_solver.train(final_iterations, inner_iterations, save_path_logs, save_path_cppweights)
    
    np.savetxt(f"{args.run_name}_training_history.csv", training_history, delimiter = ',')

if __name__ == '__main__':
    main()