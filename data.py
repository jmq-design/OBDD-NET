import torch
import math
from torch.utils.data import Dataset, DataLoader
import os
# import json
import pandas as pd
from multiprocessing import Pool
import argparse
import time

import myconfig



class AssignmentDataset(Dataset):
    def __init__(self, dataset_dir, device):
        self.atom_description = pd.read_csv(dataset_dir, nrows=0).columns.values.tolist()[:-1] # get the header (atom_description)
        raw_dataset = pd.read_csv(dataset_dir, header=0).values  # start from line 0
        dataset = torch.tensor(raw_dataset, dtype=torch.float).to(device)
        self.data = dataset[:,:-1]
        self.labels = dataset[:,-1]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_atom_description(self):
        return self.atom_description

    def get_atom_size(self):
        return len(self.data[0])
    
    def get_class_distribution(self):
        return self.labels.sum()/len(self.labels)

    def size(self):
        return len(self.data)
        
    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels





    

