import torch.nn.init
from torch import nn
from 我的路径 import 路径


class 三维卷积(nn.Module):
    def __init__(self, 分类数, 是否已训练=False):
        super(三维卷积, self).__init__()
        self.卷积1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.池化1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.卷积2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.池化2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.卷积3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.卷积3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.池化3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.卷积4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.卷积4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.池化4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.卷积5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.卷积5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.池化5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.全链接6 = nn.Linear(8192, 4096)
        self.全链接7 = nn.Linear(4096, 4096)
        self.全链接8 = nn.Linear(4096, 分类数)

        self.失活 = nn.Dropout(p=0.5)
        self.线性整流函数 = nn.ReLU()

        self.__初始化权重()

        if 是否已训练:
            self.__载入已训练的权重()

    def forward(self, 输入):
        输入 = self.线性整流函数(self.卷积1(输入))
        输入 = self.池化1(输入)

        输入 = self.线性整流函数(self.卷积2(输入))
        输入 = self.池化2(输入)

        输入 = self.线性整流函数(self.卷积3a(输入))
        输入 = self.线性整流函数(self.卷积3b(输入))
        输入 = self.池化3(输入)

        输入 = self.线性整流函数(self.卷积4a(输入))
        输入 = self.线性整流函数(self.卷积4b(输入))
        输入 = self.池化4(输入)

        输入 = self.线性整流函数(self.卷积5a(输入))
        输入 = self.线性整流函数(self.卷积5b(输入))
        输入 = self.池化5(输入)

        输入 = 输入.view(-1, 8192)
        输入 = self.线性整流函数(self.全链接6(输入))
        输入 = self.失活(输入)
        输入 = self.线性整流函数(self.全链接7(输入))
        输入 = self.失活(输入)

        对数 = self.全链接8(输入)
        return 对数

    def __初始化权重(self):
        for 模型 in self.modules():
            if isinstance(模型, nn.Conv3d):
                torch.nn.init.kaiming_normal_(模型.weight)
            elif isinstance(模型, nn.BatchNorm3d):
                模型.weight.data.fill_()
                模型.bias.data.zero_()

    def __载入已训练的权重(self):
        对应名称 = {
            # 卷积1
            "features.0.weight": "卷积1.weight",
            "features.0.bias": "卷积1.bias",
            # 卷积2
            "features.3.weight": "卷积2.weight",
            "features.3.bias": "卷积2.bias",
            # 卷积3a
            "features.6.weight": "卷积3a.weight",
            "features.6.bias": "卷积3a.bias",
            # 卷积3b
            "features.8.weight": "卷积3b.weight",
            "features.8.bias": "卷积3b.bias",
            # 卷积4a
            "features.11.weight": "卷积4a.weight",
            "features.11.bias": "卷积4a.bias",
            # 卷积4b
            "features.13.weight": "卷积4b.weight",
            "features.13.bias": "卷积4b.bias",
            # 卷积5a
            "features.16.weight": "卷积5a.weight",
            "features.16.bias": "卷积5a.bias",
            # 卷积5b
            "features.18.weight": "卷积5b.weight",
            "features.18.bias": "卷积5b.bias",
            # 全链接6
            "classifier.0.weight": "全链接6.weight",
            "classifier.0.bias": "全链接6.bias",
            # 全链接7
            "classifier.3.weight": "全链接7.weight",
            "classifier.3.bias": "全链接7.bias",
        }

        路径_字典 = torch.load(路径.模型目录())
        状态_字典 = self.state_dict()
        for 名称 in 路径_字典:
            if 名称 not in 对应名称:
                continue
            状态_字典[对应名称[名称]] = 路径_字典
        self.load_state_dict(状态_字典)


def 获得1x学习率参数(模型: 三维卷积):
    b = [模型.卷积1, 模型.卷积2, 模型.卷积3a, 模型.卷积3b, 模型.卷积4a, 模型.卷积4b, 模型.卷积5a, 模型.卷积5b, 模型.全链接6, 模型.全链接7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def 获得10x学习率参数(模型: 三维卷积):
    b = [模型.全链接8]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k
