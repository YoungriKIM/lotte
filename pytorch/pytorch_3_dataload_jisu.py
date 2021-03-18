# https://sensibilityit.tistory.com/511
# 지수가 알려준 파이토치 데이터로더 글 보고 따라하기@!!

# 1-3 torchvison.transforms 를 사용한 경우

import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 데이터 불러오기 클래스 정의
class custom_dataset(dataset):
    def __init__(self, file_list, transforms=None):
        self.file_list = file_list
        self.transforms = transforms

    # 학습 데이터의 개수를 리턴
    def __len__(self):
        return len(self.file_list)
    
    # 이미지를 연다
    def __getitem__(self, idx):
        img = cv2.imread(self.file_list[idx])
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(im_rgb.transpose(1, 0, 2))

        if self.transforms is not None:
            img = self.transforms(img)

        return img

# transform 지정
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# 커스텀데이터셋 호출
file_list = 'D:/lotte/LPD_competition/gwayeon'
dataset = custom_dataset(file_list=file_list, transforms=transform)
data_loader = DataLoader