import os.path
import sys

from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from 我的路径 import 路径


class 视频数据集(Dataset):
    def __init__(self, 数据集='ucf101', 分隔符='训练', 裁剪长度=16, 是否要预处理=False):
        self.根目录, self.输出目录 = 路径.数据库目录(数据集)
        文件夹 = os.path.join(self.输出目录, 分隔符)
        self.裁剪长度 = 裁剪长度
        self.分隔符 = 分隔符

        self.图片总数 = 0

        self.重置的高 = 128
        self.重置的宽 = 171
        self.裁剪的尺寸 = 112

        if not self.检查目录():
            raise RuntimeError('数据集没有找到或者损坏。你可以从官网上下载......')

        if (self.检查预处理()) or 是否要预处理:
            print('预处理数据集{}，这个时间会很长，但它只需要执行一次。'.format(数据集))
            self.预处理()

    def 检查目录(self):
        if not os.path.exists(self.根目录):
            return False
        else:
            return True

    def 检查预处理(self):
        if not os.path.exists(self.输出目录):
            return False
        elif not os.path.exists(os.path.join(self.输出目录, '训练')):
            return False

        for i, 视频类别 in enumerate(os.listdir(os.path.join(self.输出目录, '训练'))):
            for 视频 in os.listdir(os.path.join(self.输出目录, '训练', 视频类别)):
                视频名称 = os.path.join(
                    os.path.join(self.输出目录, '训练', 视频类别, 视频),
                    sorted(os.listdir(os.path.join(self.输出目录, '训练', 视频类别, 视频)))[0]
                )
                # 为了支持中文路径而做的转换
                图片 = Image.open(视频名称)
                图片 = cv2.cvtColor(np.asarray(图片), cv2.COLOR_RGB2BGR)
                if np.shape(图片)[0] != 128 or np.shape(图片)[1] != 171:
                    return False
                else:
                    break

            if i == 10:
                break

        return True

    def 预处理(self):
        if not os.path.exists(self.输出目录):
            os.mkdir(self.输出目录)
            os.mkdir(os.path.join(self.输出目录, '训练'))
            os.mkdir(os.path.join(self.输出目录, '验证'))
            os.mkdir(os.path.join(self.输出目录, '测试'))

        for 文件 in os.listdir(self.根目录):
            文件路径 = os.path.join(self.根目录, 文件)
            视频文件列表 = [名称 for 名称 in os.listdir(文件路径)]

            训练和验证, 测试 = train_test_split(视频文件列表, test_size=0.2, random_state=42)
            训练, 验证 = train_test_split(训练和验证, test_size=0.2, random_state=42)

            训练目录 = os.path.join(self.输出目录, '训练', 文件)
            验证目录 = os.path.join(self.输出目录, '验证', 文件)
            测试目录 = os.path.join(self.输出目录, '测试', 文件)
            if not os.path.exists(训练目录):
                os.mkdir(训练目录)
            if not os.path.exists(验证目录):
                os.mkdir(验证目录)
            if not os.path.exists(测试目录):
                os.mkdir(测试目录)

            for 视频 in 训练:
                self.处理视频(视频, 文件, 训练目录)
            print('训练目录处理完成')

            for 视频 in 验证:
                self.处理视频(视频, 文件, 验证目录)
            print('验证目录处理完成')

            for 视频 in 测试:
                self.处理视频(视频, 文件, 测试目录)
            print('测试目录处理完成')

            print('全部完成')

    def 处理视频(self, 视频, 动作名称, 保存目录):
        视频文件名 = 视频.split('.')[0]
        if not os.path.exists(os.path.join(保存目录, 视频文件名)):
            os.mkdir(os.path.join(保存目录, 视频文件名))

        捕获 = cv2.VideoCapture(os.path.join(self.根目录, 动作名称, 视频))
        帧数 = int(捕获.get(cv2.CAP_PROP_FRAME_COUNT))
        帧宽 = int(捕获.get(cv2.CAP_PROP_FRAME_WIDTH))
        帧高 = int(捕获.get(cv2.CAP_PROP_FRAME_HEIGHT))

        提炼频率 = 4
        # 确保分割后的视频至少有 16 帧
        if 帧数 // 提炼频率 <= 16:
            提炼频率 -= 1
            if 帧数 // 提炼频率 <= 16:
                提炼频率 -= 1
                if 帧数 // 提炼频率 <= 16:
                    提炼频率 -= 1
        次数 = 0
        i = 0
        是否保留 = True
        while 次数 < 帧数 and 是否保留:
            是否保留, 帧 = 捕获.read()
            if 帧 is None:
                continue

            if 次数 % 提炼频率 == 0:
                if (帧高 != self.重置的高) or (帧宽 != self.重置的宽):
                    帧 = cv2.resize(帧, (self.重置的宽, self.重置的高))

                # 为了支持中文路径而做的转换
                帧 = cv2.cvtColor(帧, cv2.COLOR_BGR2RGB)
                帧 = Image.fromarray(帧)
                帧.save(os.path.join(保存目录, 视频文件名, '0000{}.jpg'.format(str(i))))

                self.图片总数 += 1
                sys.stdout.write('\r图片总数：{}'.format(self.图片总数))
                sys.stdout.flush()

                i += 1
            次数 += 1

        捕获.release()


if __name__ == '__main__':
    print('开始')
    # 数据已经处理过一次了
    训练数据 = 视频数据集(数据集='ucf101', 分隔符='测试', 裁剪长度=8, 是否要预处理=True)
