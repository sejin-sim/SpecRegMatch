import os, random
import numpy as np
import torch, glob
from torch.utils.data import Dataset
from natsort import natsorted

class LabeledDataset(Dataset):
    def __init__(self, args, split, statistics):
        super(LabeledDataset, self).__init__()

        # set car type and their paths
        full_labeled_data = glob.glob(os.path.join("./data", "_".join([split, str(args.n_fft), str(args.hop_length)]),"*", "spectrogram.pth"))
        self.files = full_labeled_data

        if split == 'train':
            self.files = reduce_dataset(data=full_labeled_data, labeled_data_amount=args.labeled_data_N, train_1_data_amount=args.labeled_train_amount, seed=args.seed)

        self.label_types = ["label"+str(i) for i in range(1,11)] + ["label_overall"]

        self.x_min = torch.stack([statistics[f"min_{i}"] for i in range(6)], dim=0).unsqueeze(2).repeat(1, 1, args.freq_range)
        self.x_range = torch.stack([(statistics[f"max_{i}"]-statistics[f"min_{i}"]) for i in range(6)], dim=0).unsqueeze(2).repeat(1, 1, args.freq_range)

        self.y_min = torch.tensor([statistics[f"min_"+label] for label in self.label_types])
        self.y_range = torch.tensor([(statistics[f"max_"+label]-statistics[f"min_"+label]) for label in self.label_types])

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = torch.load(self.files[index])
        
        feature = (torch.cat([sample[str(i)] for i in range(6)]) - self.x_min) / (self.x_range)
        target = (torch.cat([sample[label].unsqueeze(0) for label in self.label_types]) - self.y_min) / (self.y_range)
        
        return dict(feature=feature, target=target)


class UnlabeledDataset_mix(Dataset):
    def __init__(self, args, split, statistics):
        super(UnlabeledDataset_mix, self).__init__()
        
        self.args = args
        self.masking_range = args.masking_range
        self.statistics = statistics

        # set car type and their paths
        full_unlabeled_data = glob.glob(os.path.join("./data", "_".join([split, str(args.n_fft), str(args.hop_length)]),"*"))
        self.files = full_unlabeled_data

        self.x_min = torch.stack([statistics[f"min_{i}"] for i in range(6)], dim=0).unsqueeze(2).repeat(1, 1, args.freq_range)
        self.x_range = torch.stack([(statistics[f"max_{i}"]-statistics[f"min_{i}"]) for i in range(6)], dim=0).unsqueeze(2).repeat(1, 1, args.freq_range)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = torch.load(os.path.join(self.files[index], 'spectrogram.pth'))
        raw_image = torch.cat(list([sample[str(i)] for i in range(6)]))

        weak = self.weak_transform(image=raw_image).float()
        strong = self.specaug(image=raw_image).float()
        alpha = np.random.uniform(0, 1)
        mix = (weak * alpha) + (strong * (1 - alpha))
        
        weak = (weak - self.x_min) / self.x_range
        strong = (strong - self.x_min) / self.x_range
        mix = (mix - self.x_min) / self.x_range
        
        return dict(weak=weak,strong=strong,mix=mix)
        # plt.imshow(torch.flip(self.weak_transform(image=raw_image).float().permute(1, 2, 0), dims=(0,))[:,:,0], cmap='magma')
        
    def weak_transform(self, image):

        # MultiplicativeNoise
        multiplier = np.random.uniform(0.8, 1.2, (image.shape))
        return image * multiplier

    def specaug(self, image):

        p_freq = random.randint(0, 224-self.masking_range)
        p_time = random.randint(0, 224-self.masking_range)

        image[:, p_freq:p_freq+self.masking_range, :] = 0
        image[:, :, p_time:p_time+self.masking_range] = 0

        return image


def reduce_dataset(data, labeled_data_amount, train_1_data_amount, seed):
  
    exp_names = sorted(list(set(["_".join(os.path.basename(os.path.dirname(f)).split("_")[:-1]) for f in data])))
    
    random.seed(seed)
    random.shuffle(exp_names)
    exp_names = exp_names[:labeled_data_amount]
    
    if train_1_data_amount is None or labeled_data_amount != 1:
        final_dataset = natsorted([f for f in data if "_".join(os.path.basename(os.path.dirname(f)).split("_")[:-1]) in exp_names])
    else :         
        final_dataset = natsorted([f for f in data if "_".join(os.path.basename(os.path.dirname(f)).split("_")[:-1]) in exp_names])[:train_1_data_amount]
    
    print(f"{exp_names} : {len(final_dataset)}")
    return final_dataset
