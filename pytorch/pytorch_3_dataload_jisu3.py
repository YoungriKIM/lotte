# https://blog.promedius.ai/pytorch_dataloader_1/
# 지수짱~

# 성능 잘 나오는지 test 파일 만들어서 확인하기!

import torch
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision import transforms
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------------------
# ImageFolder 이용해서 이미지 불러오기
# train 불러오기!
train_imgs = ImageFolder("D:/lotte/LPD_competition/gwayeon",
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))]))

train_loader = data.DataLoader(train_imgs, batch_size=12, shuffle=True)

# train 불러오기!
test_imgs = ImageFolder("D:/lotte/LPD_competition/gwayeon_test",
                         transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))]))

test_loader = data.DataLoader(train_imgs, batch_size=4, shuffle=True)


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
    model.fc = nn.Linear(in_features=fc_feature, out_features=10)
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
    for epoch in range(2):
        
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
                print('[%d, %5d] loss: % .3f' % 
                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('===== train done =====')
    
    # 학습한 모델을 저장
    # 토치에서 저장하기: https://pytorch.org/docs/stable/notes/serialization.html
    path = 'D:/aidata/pth/torch_01.pth'
    torch.save(model.state_dict(), path)
    print('===== save done =====')

    #----------------------------------------------------------------------------
    # 전체 테스트 데이터에 대해 acc 확인
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
    # ============================


    # -----------------------------------
    # 이 파일부터 시작
    # -----------------------------------

    #----------------------------------------------------------------------------
    # 시험용 데이터 일부로 검사하기
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # 우선 이미지 출력해서 정답 확인
    
    



    