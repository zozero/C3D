from torch import nn


class 三维卷积(nn.Module):
    def __init__(self,分类数,是否预训练=False):
        super(三维卷积, self).__init__()