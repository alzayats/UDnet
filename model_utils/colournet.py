import torch.nn as nn
from model_utils.color_change import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, x):
        out1 = self.avgpool(x)
        out2 = self.conv(x)
        out3 = self.maxpool(x)
        out4 = self.conv1(x)
        out =self.conv2(out2 / (out1 + 1e-5)+self.tanh(self.conv4(x))) + self.conv3(out4 / (out3 + 1e-5)+self.tanh(self.conv5(x)))
        return out





class RGBhs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,kernel_size=1,stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self,x):
        min = -self.maxpool(-x)
        max = self.maxpool(x)
        out = (x - min) / (max - min + 1e-5)
        out = self.conv(out)
        return out

class Labhs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,kernel_size=1,stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self,x):
        x = rgb2lab(x)
        min = -self.maxpool(-x)
        max = self.maxpool(x)
        out = (x - min) / (max - min + 1e-5)
        out = self.conv(out)
        out = lab2rgb(out)
        return out





class HSIhs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        out = rgb2hsi(x)
        h, si = torch.split(out, [1, 2], dim=1)
        minsi = -self.maxpool(-si)
        maxsi = self.maxpool(si)
        si = (si - minsi) / (maxsi - minsi + 1e-5)
        si = si + self.conv(si)
        out = torch.cat((h, si), dim=1)
        out = hsi2rgb(out)
        return out




class USLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh=nn.Tanh()
        self.step1=WB()
        self.step2=RGBhs()
        self.step3=HSIhs()
        self.step4=Labhs()
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        out=self.step1(x)
        out1=self.step2(out)+self.tanh(self.conv1(out))
        out2 = self.step3(out)+self.tanh(self.conv2(out))
        out3 = self.step4(out) + self.tanh(self.conv3(out))
        out=self.conv4(out1)+self.conv5(out2)+self.conv6(out3)

        out=1-self.relu(1-self.relu(out))
        return out

if __name__ == '__main__':
    """60.42 MMac 894"""
    from ptflops import get_model_complexity_info
    net=USLN().to(device)
    x = torch.rand(1, 3, 256, 256).to(device)
    output = net(x)
    flops,params=get_model_complexity_info(net,(3,256,256))
    print(flops,params)
    print(x.shape)
    print(output.shape)
