import glob
import os.path

import torch

if __name__ == '__main__':
    设备 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("%s设备将被使用" % 设备)

    轮回数 = 101
    恢复的轮回数 = 0
    是否测试 = True
    测试间隔 = 20
    快照 = 25
    学习率 = 1e-5

    数据集 = 'ufc101'
    if 数据集 == 'hmdb51':
        分类数 = 51
    elif 数据集 == 'ufc101':
        分类数 = 101
    else:
        print("我们的数据集只有”hmdb51“和”ufc101“")
        raise NotImplementedError

    保存路径根 = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    项目文件夹名称 = os.path.dirname(os.path.abspath(__file__)).split('\\')[-1]

    if 恢复的轮回数 != 0:
        运行 = sorted(glob.glob(os.path.join(保存路径根, '运行', '运行_*')))
        运行标识 = int(运行[-1].split('_')[-1]) if 运行 else 0
    else:
        运行 = sorted(glob.glob(os.path.join(保存路径根, '运行', '运行_*')))
        运行标识 = int(运行[-1].split('_')[-1]) + 1 if 运行 else 0

    保存路径 = os.path.join(保存路径根, '运行', '运行_' + str(运行标识))
    模型名 = '三维卷积'
    保存名 = 模型名 + '-' + 数据集


    def 训练模型(数据集=数据集, 保存路径=保存路径, 分类数=分类数, 学习率=学习率, 轮回数=轮回数, 保存轮回间隔=快照, 是否测试=是否测试, 测试间隔=测试间隔):
        pass
        # 模型=
