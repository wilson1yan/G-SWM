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


class MOVi(Dataset):
    def __init__(self, root, mode, ep_len=None, sample_length=16):
        assert mode in ['train', 'val', 'test']
        if mode == 'val':
            mode = 'test'
        
        self.root = root
        self.data = h5py.File(root, 'r')
        self.mode = mode
        self.sample_length = sample_length
        
        self._images = self.data[f'{self.mode}_data']
        self._idx = self.data[f'{self.mode}_idx'][:]

        self.size = len(self._idx)
    
    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.root, 'r')
        self._images = self.data[f'{self.mode}_data']

    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        start = self._idx[idx] 
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        if end - start > self.sample_length:
            start = start + np.random.randint(low=0, high=end - start - self.sample_length)
        assert start < start + self.sample_length <= end, f'{start}, {end}'
        video = torch.tensor(self._images[start:start + self.sample_length])
        video = video.float() / 255.
        video = video.permute(0, 3, 1, 2).contiguous()
        
        return video, torch.tensor(0.)
    
