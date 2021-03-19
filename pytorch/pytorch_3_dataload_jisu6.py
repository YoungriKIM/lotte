# https://blog.promedius.ai/pytorch_dataloader_1/
# 지수짱~

# 1000개 돌리고 제출까지~ ^^

import torch
import torchvision
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2


# -----------------------------------------------------------------------------
# ImageFolder 이용해서 이미지 불러오기
# train 불러오기!
train_imgs = ImageFolder("D:/lotte/LPD_competition/train",
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))]))

train_loader = data.DataLoader(train_imgs, batch_size=12, shuffle=True)

# test 불러오기!
test_imgs = ImageFolder("D:/lotte/LPD_competition/gwayeon_test",
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))]))

test_loader = data.DataLoader(test_imgs, batch_size=4, shuffle=True)

# pred 불러오기!
# pred_imgs = ImageFolder("D:/lotte/LPD_competition/pred/test",
#                          transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))]))

# pred_loader = data.DataLoader(pred_imgs, batch_size=4, shuffle=True)

'''
# 잘 불러와졌나 확인
print(train_imgs.classes)
print(train_imgs.class_to_idx)
# ['0', '1', '2', '3']
# {'0': 0, '1': 1, '2': 2, '3': 3}
print(len(train_imgs))
print(train_imgs[0][0].shape)
print(train_imgs[1][0].shape)
# 192
# torch.Size([3, 128, 128])
# torch.Size([3, 128, 128])
# 야호!
'''

# -----------------------------------------------------------------------------
# 전이학습 이용하기
# pretrained = True > 미리 학습된 weight 가져와서 거기서 부터 시작
# pretrained = False > 랜덤한 weight 가져오고 모델 구조만 가져올거야
if __name__ == '__main__':
    import torchvision.models as models

    # 전이모델을 지정해주자
    model = models.googlenet(pretrained=True)
    # print(model)
    # 불러온 모델의 output 부분을 변경해주자 이를 fine_tunning(미세조정이라고 한다.)

    # fine_tunning (output)을 조정하자
    # model 의 fully_connected layer의 input feature 부분 출력해서 확인하기
    fc_feature = model.fc.in_features
    print('number of fully connected layer input feature: ', fc_feature)
    # number of fully connected layer input feature:  1024

    # output node 변경
    model.fc = nn.Linear(in_features=fc_feature, out_features=1000)
    print(model.fc)
    # Linear(in_features=1024, out_features=4, bias=True)

    # -----------------------------------------------------------------------------
    # 모델을 gpu로 올리기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    # device:  cuda

    model.to(device)

    # -----------------------------------------------------------------------------
    # 여기서 부터 알아서 컴파일~마무리까지

    # 컴파일 내용 정의
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 신경망 학습하기
    for epoch in range(1):
        
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # acc 추가
            loss.backward() # 모든 그라디언트들을 자동으로 계산할 수 있게 한다.
            optimizer.step()

            # 통계를 출력
            running_loss += loss.item()
            if i % 2 == 1:    # print every 2000 batch
                print('[%d, %5d] loss: % .9f' % 
                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('===== train done =====')
    
    # 학습한 모델을 저장
    # 토치에서 저장하기: https://pytorch.org/docs/stable/notes/serialization.html
    path = 'D:/aidata/pth/torch_06.pth'
    torch.save(model.state_dict(), path)
    print('===== save done =====')

    #----------------------------------------------------------------------------
    # 전체 트레인 데이터에 대해 acc 확인
    correct = 0
    total = 0
    with torch.no_grad():   # 기록 추척 및 메모리 사용을 방지하기 위해 no_grad를 사용한다.
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('final accuracy: %d %%' % (100 * correct / total))

    #----------------------------------------------------------------------------
    # 이미지 보는 함수 만들기
    import matplotlib.pyplot as plt
    
    def imshow(img):
        img = img / 2 + 0.5  # 비정규화
        npimg = img.numpy()
        print('원래 쉐잎: ', npimg.shape)
        # 원래 쉐잎:  (3, 132, 522)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        print('트랜스포스 후 쉐잎: ', np.transpose(npimg, (1, 2, 0)).shape)
        # 트랜스포스 후 쉐잎:  (132, 522, 3)
        # 왜 네 개씩 나오는지..왜 쉐잎이 저따군지...미스테리...
        plt.show()

    #----------------------------------------------------------------------------

    #---------------------------------
    makeclasses = []
    for i in range(0, 1000):
        aaa = (str(i) + ' ')
        ccc = aaa.split()
        makeclasses.append(ccc)
    classes = tuple(np.array(makeclasses).squeeze())
    #---------------------------------

    #---------------------------------------------------------------------------
    # 전체 테스트 데이터에 대해 확인
    correct = 0
    total = 0
    with torch.no_grad():   # 기록 추척 및 메모리 사용을 방지하기 위해 no_grad를 사용한다.
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('전체 테스트 데이터 acc: %d %%' % (100 * correct / total))
    # ==============================

    #----------------------------------------------------------------------------
    # 1000가지 중 어떤 것을 더 잘 분류하고 어떤 것을 못했는지 알아보자
    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze() # squeeze : 1인 차원을 제거한다.([3,1] > [3])
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(1000):
        try:
            print('%5s 의 정확도: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        except ZeroDivisionError:
            print("ZeroDivision")

    #  ----------------------------------------------------------------------------------------------


# ================================
# 전체 테스트 데이터 acc: 95 %      > 10개일 때

# 100개일 떄 pytorch_3_dataload_jisu4
# 에폭1
# final accuracy: 91 %
# 전체 테스트 데이터 acc: 12 %


#  ----------------------------------------------------------------------------------------------
# 예측해서 저장하자!
import albumentations
from torch.utils.data import Dataset, DataLoader

class kakakaka(Dataset):
    def __init__(self, imgs=None, labels=None, transform=None, train=True):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train=train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get1
        img = self.imgs[idx]
        img = self.transform(img)
        
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img

# 예측 이미지 불러오기
pred_imgs_dir = np.array(sorted(glob.glob('../dacon12/data/test/*')))
# 예측 이미지만 모으기
pred_imgs=[]
for path in tqdm(pred_imgs_dir):
    pred_img=cv2.imread(path, cv2.IMREAD_COLOR)
    pred_imgs.append(pred_img)
pred_imgs=np.array(pred_imgs)

pred_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((128, 128))])


# ================ Test 추론 =============================
submission = pd.read_csv('../dacon12/data/sample_submission.csv')

with torch.no_grad():
    model.load_state_dict(torch.load('D:/aidata/pth/torch_06.pth'))
    model.eval()

    pred_dataset = kakakaka(imgs = pred_imgs, transform=pred_transform)
    pred_loader = DataLoader(dataset=pred_dataset, batch_size=32, shuffle=False)

    for n, X_test in enumerate(tqdm(pred_loader)):
        X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            pred_test = model(X_test).cpu().detach().numpy()
            submission.iloc[n*32:(n+1)*32,1:] += pred_test

# ==================== 제출물 생성 ====================
submission.iloc[:,1:] = np.where(submission.values[:,1:]>=3.0, 1,0)
submission.to_csv('D:/lotte/LPD_competition/sub/sub_01.csv', index=False)
print('===== done =====')