# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
# https://www.kaeee.de/2020/10/26/pytorch-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%A1%9C%EB%8D%94-%EB%A7%8C%EB%93%A4%EA%B8%B0.html
# 파이토치 cifar10 예제
# 한 걸로 롯데 돌려보기

import torch
import torchvision
import torchvision.transforms as transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
# skimage : scikitlearn_image , pillow보다 고급 기능
# io: 다양한 형식의 이미지를 읽고 쓰는 유틸리티
# 설명: https://aciddust.github.io/blog/post/scikit-Python-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%B2%98%EB%A6%AC-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC/


# trasform 지정
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128)), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])

# 불러올 파일 경로 지정
data_path_train = glob('../lotte_data/LPD_competition/gwayeon/*/*.jpg') # 0,1,2,3

# 클래스 지정
classes = ('0', '1', '2', '3')


# 데이터셋 정의
class Mydataset(Dataset):
    #data_path_list - 이미지 path 전체 리스트
    #label - 이미지 ground truth
    
    # 생성자인 __init__ 을 정의. 이미지의 path 리스트와, 클래스 명, transform을 받는다.
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    # get_label : 라벨을 반환받자
    def get_label(self, path_list):
        label_list = []
        for path in path_list:
            # 뒤에서 두번째가 class
            label_list.append(path.split('/')[-2])
        return label_list

    # __len__ 전체 데이터셋의 길이를 반환
    def __len__(self):
        return len(self.path_list)

    # __getitem__ : 학습에 쓸 이미지를 반환/ 이미지를 텐서형태의 인풋으로>리스트로 변환 > 클래스를 숫자로 반환
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


# Dataloader 세팅
trainloader = torch.utils.data.DataLoader(Mydataset(data_path_train, classes, transform=transform), batch_size=4, shuffle=True)
#---------------------------------------------------------------------------
# 이미지 보고 확인하기
# if __name__ == '__main__':
                
# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답(label) 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
