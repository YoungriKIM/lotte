# http://blog.naver.com/PostView.nhn?blogId=ckdgus1433&logNo=221407059199&parentCategoryNo=&categoryNo=11&viewDate=&isShowPopularPosts=true&from=search


# 라이브러리 불러오기

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

# 학습을 위한 데이터 증가(Augmentation)와 일반화하기
# 단지 검증을 위한 일반화하기
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'D:/lotte/LPD_competition'

path = {x: os.path.join(os.path.dirname(os.path.abspath(__file__)),data_dir,x)
                for x in ['train', 'val']}

# path['train']은 train set의 경로
# path['val']은 val set의 경로
# join은 문자열을 이어 붙여주는 함수

image_datasets = {x: datasets.ImageFolder(path[x],
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = { 'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                                             shuffle=True, num_workers=0),
                'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                             shuffle=True, num_workers=0) }

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#dataset_sizes['train'] = train set 사진 갯수
#dataset_sizes['val'] = val set 사진 갯수

class_names = image_datasets['train'].classes
#class_names = ['ants', 'bees']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#GPU가 이용 가능한지를 확인
