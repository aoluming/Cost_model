import torch
import torch.nn as nn
from torch.nn import functional as F

class attentionblock(nn.Module):
    def __init__(self,in_channel):
        super(attentionblock,self).__init__()
        self.pool=nn.AdaptiveMaxPool3d(1)
        self.in_channel=in_channel
        self.conv=nn.Conv3d(self.in_channel,self.in_channel,kernel_size=(1,1,1))
        self.linear=nn.Linear(self.in_channel*3,self.in_channel*3)
    def forward(self,input1,input2,input3):
        input1=self.pool(input1)
        input2=self.pool(input2)
        input3=self.pool(input3)
        input1=self.conv(input1)
        input2 = self.conv(input2)
        input3 = self.conv(input3)
        input1=input1.squeeze()
        input2 = input2.squeeze()
        input3 = input3.squeeze()
        a=torch.cat((input1,input2,input3),1)
        a=self.linear(a)
        a=F.softmax(a,dim=0)
        return a
class costblock(nn.Module):
    def __init__(self,in_channel,channel,stride=1):
        super(costblock,self).__init__()
        self.stride=stride
        self.in_channel=in_channel
        self.channel=channel
        self.conv1=nn.Conv3d(self.in_channel,self.channel,kernel_size=(1,1,1))
        self.conv=nn.Conv2d(self.channel,self.channel,kernel_size=(3,3),padding=(1,1),stride=(self.stride,self.stride))
        self.conv2=nn.Conv3d(self.channel,self.channel*4,kernel_size=(1,1,1))
        self.attenblock=attentionblock(self.channel)
        self.relu=nn.ReLU(inplace=True)
        self.batchnorm=nn.BatchNorm2d(self.channel)
    def forward(self, input):

        input=self.conv1(input)
        x1=input.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3]*input.shape[4])
        x2=input.transpose(2,3)
        x2=x2.contiguous().view(x2.shape[0],x2.shape[1],x2.shape[2],x2.shape[3]*x2.shape[4])
        x3=input.transpose(2,4)
        x3 = x3.contiguous().view(x3.shape[0], x3.shape[1], x3.shape[2], x3.shape[3] * x3.shape[4])
        out1=self.conv(x1)
        out1=self.batchnorm(out1)
        out1=self.relu(out1)
        out1=out1.view(input.shape[0],input.shape[1],input.shape[2],int(input.shape[3]/self.stride),int(input.shape[4]/self.stride))
        out2=self.conv(x2)
        out2=self.batchnorm(out2)
        out2=self.relu(out2)
        out2 = out2.view(input.shape[0], input.shape[1], input.shape[2], int(input.shape[3] / self.stride),int(input.shape[4] / self.stride))
        out3=self.conv(x3)
        out3=self.batchnorm(out3)
        out3=self.relu(out3)
        out3 = out3.view(input.shape[0], input.shape[1], input.shape[2], int(input.shape[3] / self.stride),int(input.shape[4] / self.stride))
        a=self.attenblock(out1,out2,out3)
        a1,a2,a3=a.chunk(3,dim=1)
        output1=out1.permute(2,3,4,0,1)*a1+out2.permute(2,3,4,0,1)*a2+out3.permute(2,3,4,0,1)*a3
        output1=output1.permute(3,4,0,1,2)
        output1=self.conv2(output1)
        return output1

class Cost(nn.Module):
    """
    The Cost network.
    """

    def __init__(self, num_classes,block, layers,pretrained=False):
        super(Cost, self).__init__()

        self.in_channels=64
        self.conv1=nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.maxpool2=nn.MaxPool3d(kernel_size=(3,1,1),stride=(2,1,1),padding=(1,0,0))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.layer1(out)
        out=self.maxpool2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    def _make_layer(self, block, channels, n_blocks, stride=1):
        assert n_blocks > 0, "number of blocks should be greater than zero"
        layers = []
        layers.append(block(self.in_channels, channels, stride))
        self.in_channels = channels * 4
        for i in range(1, n_blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 224, 224)
    net=Cost(101,costblock, [3,4,6,3])
    #testing for res50 , change the list for res18 or res101
    #net = costblock(64,64,stride=2)

    outputs = net(inputs)
    print(outputs.size())
