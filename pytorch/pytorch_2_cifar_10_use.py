# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
# 파이토치 cifar10 예제
# 한 걸로 롯데 돌려보기

# CIFAR10를 불러오고 정규화하기
import torch
import torchvision
import torchvision.transforms as transforms

# gpu에서 학습되도록 세팅
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)   # cuda:0

if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------
    # 데이터 처리 형태 지정
    transform = transforms.Compose([transforms.ToTensor(), rescale(128), transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])
    # transforms.ToTensor(): 데이터 타입을 Tensor 형태로 변경
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) : 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 하지만 ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜.
    # 위 코드를 이용하여 -1 ~ 1사이의 값으로 normalized 시킴
