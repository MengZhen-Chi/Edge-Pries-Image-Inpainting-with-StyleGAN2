import os
from torch.utils.data import Dataset
from PIL import Image
from models.psp import pSp
import torchvision.transforms as transforms
import sys
import random
sys.path.append('/home/zhr/test/Rec/dataset')
import data_utils
from add_edge import add_edge_toImage, add_edge_toImage_edit, add_edge_toImage_edit_without

class ImagesDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        if opts.test:
            self.source_paths = sorted(data_utils.make_dataset(opts.test_source_root))
            self.mask_path = sorted(data_utils.make_dataset(opts.test_mask_root))
            self.edge_root = sorted(data_utils.make_dataset(opts.test_edge_root))
        else:
            self.source_paths = sorted(data_utils.make_dataset(opts.source_root))
            self.mask_path = sorted(data_utils.make_dataset(opts.mask_root))
            self.edge_root = sorted(data_utils.make_dataset(opts.edge_root))    
        
    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        edge_path = self.edge_root[index]
        mask_path = self.mask_path[index]
        image_edge, ori_image, mask = add_edge_toImage_edit(from_path, edge_path, mask_path)

        return image_edge, ori_image, mask, len(self.source_paths)
    