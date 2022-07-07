import os.path

from torch.utils.data import Dataset
from 我的路径 import 路径


class 视频数据集(Dataset):
    def __init__(self, 数据集='ufc101', 分隔符='训练', 裁剪长度=16, 是否是预处理=False):
        self.根目录, self.输出目录 = 路径.数据库目录(数据集)
        文件夹 = os.path.join(self.输出目录, 分隔符)
        self.裁剪长度 = 裁剪长度
        self.分隔符 = 分隔符
