# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
# 파이토치 cifar10 예제

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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])
    # transforms.ToTensor(): 데이터 타입을 Tensor 형태로 변경
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) : 이미지의 경우 픽셀 값 하나는 0 ~ 255 값을 갖는다. 하지만 ToTensor()로 타입 변경시 0 ~ 1 사이의 값으로 바뀜.
    # 위 코드를 이용하여 -1 ~ 1사이의 값으로 normalized 시킴

    # 테스트 데이터 저장하고 경로 지정
    trainset = torchvision.datasets.CIFAR10(root='D:/aidata/cifar10', train=True, download=True,
                                            transform=transform)
    # root : 경로 지정
    # train : train or test 데이터를 받아옴.
    # transorm 우리가 사전에 설정해 놓은 데이터 처리 형태
    # download 데이터 셋이 없을때.

    # 저장한 경로의 데이터를 데이터로 쓰겠다
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    # dataset : 우리가 불러올 데이터 셋
    # batch_size = batch 단위 만큼 데이터를 뽑아옴.
    # shuffle : 데이터를 shuffle할 것인지.

    # 테스트에도 적용
    testset = torchvision.datasets.CIFAR10(root='D:/aidata/cifar10', train=False, download=True,
                                            transform=transform)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #----------------------------------------------------------------------------
    # 이미지 보기 
    if __name__ == '__main__':
                    
        import matplotlib.pyplot as plt
        import numpy as np

        # 이미지를 보여주기 위한 함수

        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()


        # 학습용 이미지를 무작위로 가져오기
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
    '''
        # 이미지 보여주기
        imshow(torchvision.utils.make_grid(images))
        # 정답(label) 출력
        print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    '''
    #----------------------------------------------------------------------------
    # 에러 대응, 실행하는 곳이 메인일 경우 
    # if __name__ == '__main__':
    #                 freeze_support()
    #                 ...


    # 합성곱 신경망(Convolution Neural Network) 정의하기
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net,self).__init__()
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
            self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
            self.fc1 = nn.Linear(in_features= 16*5*5, out_features=120)
            self.fc2 = nn.Linear(in_features= 120, out_features=84)
            self.fc3 = nn.Linear(in_features= 84, out_features=10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

            
    net = Net()
    net.to(device)

        
    # 컴파일 내용 정의
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    # criterion : 기준 즉 los
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    # 신경망 학습하기
    for epoch in range(1):
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후
            inputs, labels = data[0].to(device), data[1].to(device)

            # 변화도(gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)

            # ----------------------------
            # 확인용
            # print('in:\n',outputs)
            # print('in:\n',len(outputs))
            # print('in:\n',outputs.shape)
            # # ----------------------------
            # print(outputs)
            # print(labels)
            # ----------------------------

            loss = criterion(outputs, labels)
            loss.backward() # 모든 그라디언트들을 자동으로 계산할 수 있게 한다.
            optimizer.step()

            # 통계를 출력
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 batch
                print('[%d, %5d] loss: % .3f' % 
                (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('===== done =====')

    # 학습한 모델을 저장
    # 토치에서 저장하기: https://pytorch.org/docs/stable/notes/serialization.html
    path = 'C:/data/h5/cifar_torch.pth'
    torch.save(net.state_dict(), path)

    #----------------------------------------------------------------------------
    # 시험용 데이터로 신경망 검사하기(테스트의 일부만!)
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # 우선 이미지를 출력해서 정답 확인
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    # 모델이 예측한 값 확인
    images = images.to(device)  # 메모리를 gpu로 돌리기
    outputs = net(images)
    
    # 받은 인덱스의 가장 높은 값으로 확인
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    '''
    #----------------------------------------------------------------------------
    # 저장했던 모델을 불러오기(이 파일에서는 필요없지만)
    net = Net()
    net.load_state_dict(torch.load(path))
    '''
    
    #----------------------------------------------------------------------------
    # 전체 테스트 데이터에 대해 확인
    correct = 0
    total = 0
    with torch.no_grad():   # 기록 추척 및 메모리 사용을 방지하기 위해 no_grad를 사용한다.
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('final accuracy: %d %%' % (100 * correct / total))
    # ============================

    #----------------------------------------------------------------------------
    # 10가지 중 어떤 것을 더 잘 분류하고 어떤 것을 못했는지 알아보자
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            print('확인해!: ', labels)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze() # squeeze : 1인 차원을 제거한다.([3,1] > [3])
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        print('%5s 의 정확도: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        

        # 199줄 왜 라벨즈가 4가 최대인데 ? ㅡㅡ