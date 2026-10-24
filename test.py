import torch
import torch.nn as nn

class simplecnn(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),#3通道，16通道，3*3卷积核，1步长，1填充，保持图像大小不变 224->224
            nn.ReLU(),#激活函数，将负数映射为0，将正数保持不变，增加一些非线性特征
            nn.MaxPool2d(kernel_size=2,stride=2),#池化层，2*2池化核，2步长，将图像大小减半 112->112
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),#16通道，32通道，3*3卷积核，1步长，1填充，保持图像大小不变 112->112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) #32通道，64通道，2*2池化核，2步长，将图像大小减半 112->56
        )
        self.classifier = nn.Sequential(#分类头
            nn.Linear(32*56*56,128),#全连接层，输入32*56*56个神经元，输出128个神经元
            nn.ReLU(),
            nn.Linear(128,num_classes)#num_classes是我们的分类数
        )
    def forward(self,x):
        x = self.feature(x)#将图像输入到特征提取层进行特征提取
        x = x.view(x.size(0),-1)#展平为一维，x.size(0)为batch_size大小
        x = self.classifier(x)
        return x

model = simplecnn(num_classes = 10)#对模型进行实例化
input = torch.randn(64,3,224,224)
output = model(input)
print(output.shape)