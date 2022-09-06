import torch
import os,glob
import  random,csv

from torch.utils.data import Dataset



class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        self.root =root
        self.resize =resize

        self.name2label = {}
        for name in os.listdir(os.path.join(root)):
            if not sorted(os.path.isdir(os.path.join(root,name)))
                continue
            self.name2label[name] =len(self.name2label.keys())

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

