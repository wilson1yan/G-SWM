from torch.utils.data import Dataset
import h5py
from torchvision import transforms
import glob
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Rot(Dataset):
    def __init__(self, root, mode, ep_len=41, sample_length=10):
        assert mode in ['train', 'val', 'test']
        self.root = root
        file = os.path.join(self.root, f'{mode}.hdf5')
        assert os.path.exists(file), 'Path {} does not exist'.format(file)
        self.file = file

        self.mode = mode
        self.sample_length = sample_length
        
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        
    def __getitem__(self, index):
        
        
        with h5py.File(self.file, 'r') as f:
            self.imgs = f['imgs']
            self.actions = f['actions']
        
            if self.mode == 'train':
                # Implement continuous indexing
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length
                img = self.imgs[ep][offset:end]
                action = self.actions[ep][offset:end]
            else:
                img = self.imgs[index]
                action = self.actions[index]
                
                assert img.shape[0] == self.EP_LEN
        
        
        img = torch.from_numpy(img).permute(0, 3, 1, 2)
        img = img.float() / 255.0
        
        return img, action
    
    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            length = len(f['imgs'])
            if self.mode == 'train':
                return length * self.seq_per_episode
            else:
                return length
    

