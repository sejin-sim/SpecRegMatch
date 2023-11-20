import os, torch, glob
from tqdm import tqdm
from itertools import product
import torch.nn.functional as F

import sys
sys.path.append("")

from utils.args import PROPOSED_PARSER
from dataloaders.datasets.dataset import reduce_dataset

def calculate_statistics(args):
    
    full_labeled_data = glob.glob(os.path.join("./data", "_".join(["train", str(args.n_fft), str(args.hop_length)]),"*",'spectrogram.pth'))
    labeled_data = reduce_dataset(data=full_labeled_data, labeled_data_amount=args.labeled_data_N, train_1_data_amount=args.labeled_train_amount, seed=args.seed)
    full_unlabeled_data = glob.glob(os.path.join("./data", "_".join(["unlabeled", str(args.n_fft), str(args.hop_length)]),"*",'spectrogram.pth'))
    
    # labeled
    labeled_statistics = {}
    features, labels = [], []
    features_max, features_min = [], []
    label_types = [f"label{i}" for i in range(1,11)] + ['label_overall']
    
    for samples in tqdm(labeled_data):
        samples = torch.load(samples)
        labels.append(torch.cat([samples[label].unsqueeze(0).cuda(args.cuda) for label in label_types],axis=0).unsqueeze(0).cuda(args.cuda))
        features_max.append(torch.stack([samples[str(c)].max(dim=-1)[0].cuda(args.cuda) for c in range(6)], dim=1))
        features_min.append(torch.stack([samples[str(c)].min(dim=-1)[0].cuda(args.cuda) for c in range(6)], dim=1))

    labels = torch.cat(labels,axis=0).cuda(args.cuda)
    features_max = torch.cat(features_max).cuda(args.cuda)
    features_min = torch.cat(features_min).cuda(args.cuda)

    labeled_statistics.update({"corr": (labels.T).corrcoef() })  
    labeled_statistics.update({"similar": torch.tensor([[F.cosine_similarity(labels[:, i], labels[:, j], dim=0) for j in range(11)] for i in range(11)]) })  

    labeled_statistics.update({("min"+f"_{key}"):value for key, value in zip(label_types, labels.min(axis=0)[0].cpu().numpy())})
    labeled_statistics.update({("max"+f"_{key}"):value for key, value in zip(label_types, labels.max(axis=0)[0].cpu().numpy())})
    labeled_statistics.update({(f"max_{str(key)}"):value for key, value in zip(range(6), features_max.max(dim=0)[0].cpu())})
    labeled_statistics.update({(f"min_{str(key)}"):value for key, value in zip(range(6), features_min.min(dim=0)[0].cpu())})
    
    # for UCVME   
    for n, l in zip(label_types, labels.T):     
        scaled_label = (l - labeled_statistics['min'+f"_{n}"]) / (labeled_statistics['max'+f"_{n}"]-labeled_statistics['min'+f"_{n}"])
        labeled_statistics.update({f'scaled_mean_{n}' : scaled_label.mean().cpu().numpy().item()})
        labeled_statistics.update({f'scaled_std_{n}' : scaled_label.std().cpu().numpy().item()})
    
    # unlabeled
    unlabeled_statistics = {}
    features_max, features_min = [], []
    
    for samples in tqdm(full_unlabeled_data):
        samples = torch.load(samples)
        features_max.append(torch.stack([samples[str(c)].max(dim=-1)[0].cuda(args.cuda) for c in range(6)], dim=1))
        features_min.append(torch.stack([samples[str(c)].min(dim=-1)[0].cuda(args.cuda) for c in range(6)], dim=1))
            
    features_max = torch.cat(features_max).cuda(args.cuda)
    features_min = torch.cat(features_min).cuda(args.cuda)
    
    unlabeled_statistics.update({(f"max_{str(key)}"):value for key, value in zip(range(6), features_max.max(dim=0)[0].cpu())})
    unlabeled_statistics.update({(f"min_{str(key)}"):value for key, value in zip(range(6), features_min.min(dim=0)[0].cpu())})
    
    torch.save(labeled_statistics, os.path.join("./data", f'labeled_statistics_{args.labeled_data_N}_{args.labeled_train_amount}.pth'))
    torch.save(unlabeled_statistics, os.path.join("./data", 'unlabeled_statistics.pth'))
    
    return 3
    
    
# def main():
#     num_exps = [24 ] # , 2, 6, 12, 24]
#     labeled_train_amounts = [None] # 100, 200]
    
#     params = list(product(num_exps, labeled_train_amounts))
#     for param in params:
#         # statics = torch.load(os.path.join("./data", f'statistics_{param[0]}_{param[1]}.pth'))
#         # print(statics)
#         parser = PROPOSED_PARSER() 
#         args = parser.parse_args()

#         args.labeled_data_N = param[0]
#         args.labeled_train_amount = param[1]
        
#         calculate_statistics(args)
 
# if __name__=="__main__":
#     main()
        