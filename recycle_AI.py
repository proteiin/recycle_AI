#install and import library
#!pip install torch
#!pip install torchvision

pip install torch
pip install torchvision

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader


# set hyperparameter
batch_size = 10
learning_rate = 0.0002
num_epoch = 10

#Creating dataset

import torch
import torch.utils.data as data

class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__ == "__main__":
    train_x = torch.rand(500)
    train_y = torch.rand(500)
    tr_dataset = BasicDataset(train_x, train_y)


'train_data ='
'test_data ='

#dataloader

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
#dataloader에 [image,label] 형태로 담기도록
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)



#Building CNN model
class CNN(nn.Module):
    def __init__(self):
    	# super함수는 CNN class의 부모 class인 nn.Module을 초기화
        super(CNN, self).__init__()

        # batch_size = 10
        self.layer = nn.Sequential(
            # [100,1,28,28] -> [100,16,24,24]
            # 흑백이니 channel이 한개(수정할 것)
            # 실제는 [100,3,28,28] -> [100,16,24,24]의 형식으로 추정
            # [배치 크기, 채널, 높이, 너비]
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5),
            nn.ReLU(),

            # [100,16,24,24] -> [100,32,20,20]
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5),
            nn.ReLU(),

            # [100,32,20,20] -> [100,32,10,10]
            nn.MaxPool2d(kernel_size=2,stride=2),

            # [100,32,10,10] -> [100,64,6,6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),

            # [100,64,6,6] -> [100,64,3,3]
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layer = nn.Sequential(
        	# [100,64*3*3] -> [100,100]
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            # [100,100] -> [100,10]
            nn.Linear(100,10)
        )
        # 예제에서 가져온 코드의 data인 mnist dataset은 0~9의 숫자를 추출하기 때문에 one-hot encoding으로 10개짜리 list로 이루어져 결과물이 10개
        # 우리는 0-99의 숫자를 출력할 것이므로 결과값을 100으로 수정할것임

    def forward(self,x):
    	# self.layer에 정의한 연산 수행
        out = self.layer(x)
        # view 함수를 이용해 텐서의 형태를 [100,나머지]로 변환
        out = out.view(batch_size,-1)
        # self.fc_layer 정의한 연산 수행
        out = self.fc_layer(out)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training dataset
loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y= label.to(device)

        optimizer.zero_grad()

        output = model.forward(x)

        loss = loss_func(output,y)
        loss.backward()
        optimizer.step()

        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())


correct = 0
total = 0

# evaluate model
model.eval()

with torch.no_grad():
    for image,label in test_loader:
        x = image.to(device)
        y= label.to(device)

        output = model.forward(x)

        # torch.max함수는 (최댓값,index)를 반환
        _,output_index = torch.max(output,1)

        # 전체 개수 += 라벨의 개수
        total += label.size(0)

        # 도출한 모델의 index와 라벨이 일치하면 correct에 개수 추가
        correct += (output_index == y).sum().float()

    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))


##git 작동 테스트용 코드
