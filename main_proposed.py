import os, torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

from utils.args import PROPOSED_PARSER
from task.proposed import Trainer

import gc
gc.collect()
torch.cuda.empty_cache()

def plot_evaluation_metircs(pth: str,save_type: str,trains: list, vals: list, tests: list):

    plt.figure(figsize=(20, 8))
    plt.plot(np.arange(1, len(trains)+1, 1), trains, c='blue', label='Train', marker="v")
    plt.plot(np.arange(1, len(trains)+1, 1), vals, c='green', label='Valid', marker="s")
    plt.plot(np.arange(1, len(trains)+1, 1), tests, c='red', label='Test', marker="*")
    plt.legend()
    plt.savefig(os.path.join(pth, f'plot_{save_type}.png'))
    plt.close()


def main():
    
    gc.collect()
    torch.cuda.empty_cache()
    
    num_exps = [1] # , 2, 6, 12, 24]
    labeled_train_amounts = [None] 
    lambd_bts = [0.0001]
    lambda_ts = [0.005]
    lambda_mxs = [0.1] 
    params = list(product(num_exps, labeled_train_amounts, lambd_bts, lambda_ts, lambda_mxs))


    for param in params:
        
        parser = PROPOSED_PARSER() 
        args = parser.parse_args()
        
        args.epochs = 1
        
        args.labeled_data_N = param[0]
        args.labeled_train_amount = param[1]
        args.lambd_bt = param[2]
        args.lambda_t = param[3]
        
        args.lambda_mx = param[4]
        args.lambda_mix = 0.5
               
        args.result_name = f"Flops"

        print("labeled_data_N=", args.labeled_data_N)
        
        trainer = Trainer(args)
        flag_loss = np.inf       
        
        time_memory = []
        for epoch in range(1, args.epochs+1, 1):
            with torch.autograd.profiler.profile(use_cpu=True, use_cuda=True, profile_memory=True, with_flops=True) as prof:
                trainer.training(epoch)
            total_average = pd.Series(vars(prof.total_average()))
            total_average['labeled_data_N'] = args.labeled_data_N
            time_memory.append(total_average)
            
            df = pd.DataFrame(time_memory)
            df.to_csv(f"{parser.prog}_{args.labeled_data_N}.csv", index=False)
                
            # s, val_loss = trainer.evaluation(epoch, 'Valid', save=False)
                            
            # if val_loss < flag_loss:
            #     _ = trainer.evaluation(epoch, 'Test', save=s)
            #     flag_loss = val_loss
        

        

        
if __name__=="__main__":
    # for num_exp in [1, 2, 6, 12, 24]:
    main()
    
    