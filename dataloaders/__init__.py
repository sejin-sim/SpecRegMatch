from dataloaders.datasets.dataset import LabeledDataset, UnlabeledDataset_mix
import torch
from torch.utils.data import DataLoader, Sampler

# data loader for semi-supervised learning ( unlabeled >> labeled ) 
class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_loaders(args, labeled_statistics, unlabeled_statistics, num_workers=0):

    train_set = LabeledDataset(args, 'train', labeled_statistics)
    valid_set = LabeledDataset(args, 'val', labeled_statistics)
    test_set = LabeledDataset(args, 'test', labeled_statistics)
    unlabel_set = UnlabeledDataset_mix(args, 'unlabeled', unlabeled_statistics)
    
    train_loader = DataLoader(train_set,batch_size=args.batch_size // 2,
                              drop_last=True,sampler=RandomSampler(len(train_set), (len(unlabel_set) // (args.batch_size)) * (args.batch_size)),
                              num_workers=num_workers)

    valid_loader = DataLoader(valid_set,batch_size=args.batch_size,num_workers=num_workers,shuffle=False)

    test_loader = DataLoader(test_set,batch_size=args.batch_size,num_workers=num_workers,shuffle=False)
   
    unlabel_loader = DataLoader(unlabel_set, batch_size=args.batch_size // 2, drop_last=True,
                                sampler=RandomSampler(len(unlabel_set), (len(unlabel_set) // (args.batch_size)) * (args.batch_size)),
                                num_workers=num_workers
                                )
    
    return train_loader, valid_loader, test_loader, unlabel_loader
