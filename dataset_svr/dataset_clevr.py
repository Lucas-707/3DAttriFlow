import torch.utils.data as data
import os.path
import torch
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import pickle
from os.path import join, dirname, exists
from easydict import EasyDict
import json
from termcolor import colored
import dataset_svr.pointcloud_processor as pointcloud_processor
from copy import deepcopy
import yaml
import random

def red_print(x):
    print(colored(x, "red"))

class CLEVR_ft(data.Dataset):

    def __init__(self, opt, train=True):
        self.opt = opt
        self.num_sample = opt.number_points
        self.train = train
        self.init_normalization()
        self.init_singleview()

        self.img_dir = "/home/yuwu3/SpaceQA/Code/Preprocess/clevr-dataset-gen/output/images"
        self.pc_dir = "/home/yuwu3/SpaceQA/Code/Preprocess/clevr-dataset-gen/output/points"
        
        self.datapath = []
        if self.train:
            self.indices = list(range(450))
        else:
            self.indices = list(range(450, 500))
        for i in self.indices:
            img_path = os.path.join(self.img_dir, f"CLEVR_new_{i:06d}.png")
            pc_path = os.path.join(self.pc_dir, f"CLEVR_new_{i:06d}.npy")
            self.datapath.append( (img_path, pc_path) )

    def init_normalization(self):
        if self.opt.normalization == "UnitBall":
            self.normalization_function = pointcloud_processor.Normalization.normalize_unitL2ball_functional
        elif self.opt.normalization == "BoundingBox":
            self.normalization_function = pointcloud_processor.Normalization.normalize_bounding_box_functional
        else:
            self.normalization_function = pointcloud_processor.Normalization.identity_functional

    def init_singleview(self):
        ## Define Image Transforms
        self.transforms = transforms.Compose([
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor(),
        ])
        self.dataAugmentation = transforms.Compose([
            transforms.RandomCrop(127),
        ])

    def __len__(self,):
        return len(self.datapath)
    
    def __getitem__(self, index):
        return_dict = {}
        img_path, pc_path = self.datapath[index]
        img = Image.open(img_path)
        rand_size = random.randint(55, 64)
        rand_center_crop = transforms.CenterCrop(rand_size)
        img = rand_center_crop(img)
        img = self.transforms(img)[:3]
        # print("img shape after transform: ", img.shape)
        # if self.train:
        #     img = self.dataAugmentation(img)
        #     print("img shape after aug: ", img.shape)

        pc = torch.load(pc_path)
        pc = self.normalization_function(pc).squeeze().contiguous()

        return_dict['image'] = img
        return_dict['points'] = pc
        return_dict['pointcloud_path'] = pc_path
        return return_dict
    



def build_dataset(args):
    # Create Datasets
    dataset_train = CLEVR_ft(args, train=True)
    dataset_test = CLEVR_ft(args, train=False)

    # Create dataloaders
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=args.batch_size,
                                                                shuffle=False, num_workers=int(args.workers))

    len_dataset = len(dataset_train)
    len_dataset_test = len(dataset_test)
    print('Length of train dataset:%d', len_dataset)
    print('Length of test dataset:%d', len_dataset_test)

    return dataloader_train, dataloader_test

def build_dataset_val(args):

    # Create Datasets
    dataset_test = CLEVR_ft(args, train=False)

    # Create dataloaders
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                                batch_size=args.batch_size,
                                                                shuffle=False, num_workers=int(args.workers))

    len_dataset_test = len(dataset_test)
    print('Length of test dataset:%d', len_dataset_test)

    return dataloader_test