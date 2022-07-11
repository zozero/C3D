import glob
import os.path
import socket
import timeit
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from 数据载入器.数据集 import 视频数据集
from 网络 import 三维卷积模型类

if __name__ == '__main__':
    设备 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("%s设备将被使用" % 设备)

    轮回数 = 101
    恢复的轮回数 = 0
    是否测试 = True
    测试间隔 = 20
    快照 = 25
    学习率 = 1e-5

    数据集 = 'ucf101'
    if 数据集 == 'hmdb51':
        分类数 = 51
    elif 数据集 == 'ucf101':
        分类数 = 101
    else:
        print("我们的数据集只有”hmdb51“和”ucf101“")
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
        模型 = 三维卷积模型类.三维卷积(分类数=分类数, 是否已训练=False)
        训练用参数 = [
            {'params': 三维卷积模型类.获得1x学习率参数(模型), 'lr': 学习率},
            {'params': 三维卷积模型类.获得10x学习率参数(模型), 'lr': 学习率}
        ]
        准绳 = nn.CrossEntropyLoss()
        优化器 = optim.SGD(训练用参数, lr=学习率, momentum=0.9, weight_decay=5e-4)
        调度器 = optim.lr_scheduler.StepLR(优化器, step_size=10, gamma=0.1)

        if 恢复的轮回数 == 0:
            print("训练{}".format(模型名))
        else:
            检查点 = torch.load(os.path.join(保存路径, '模型', 保存名 + '_轮回-' + str(恢复的轮回数 - 1) + '.pth.tar'),
                             map_location=lambda storage, loc: storage)
            print("从{}初始权重".format(os.path.join(保存路径, '模型', 保存名 + '_轮回-' + str(恢复的轮回数 - 1) + '.pth.tar')))
            模型.load_state_dict(检查点['state_dict'])
            优化器.load_state_dict(检查点['opt_dict'])

        print('全部参数：%.2fM' % (sum(p.numel() for p in 模型.parameters()) / 1000000.0))
        模型.to(设备)
        准绳.to(设备)

        日志目录 = os.path.join(保存路径, '模型', datetime.now().strftime("%b%d_%H-%M-%S") + '_' + socket.gethostname())
        作者 = SummaryWriter(log_dir=日志目录)

        print('数据集{}训练模型......'.format(数据集))
        训练用数据载入器 = DataLoader(视频数据集(数据集=数据集, 分隔符='训练', 裁剪长度=16), batch_size=6, shuffle=True, num_workers=0)
        验证用数据载入器 = DataLoader(视频数据集(数据集=数据集, 分隔符='验证', 裁剪长度=16), batch_size=6, num_workers=0)
        测试用数据载入器 = DataLoader(视频数据集(数据集=数据集, 分隔符='测试', 裁剪长度=16), batch_size=6, num_workers=0)

        训练验证_载入器 = {'训练': 训练用数据载入器, '验证': 验证用数据载入器}
        训练验证_数量 = {x: len(训练验证_载入器[x].dataset) for x in ['训练', '验证']}
        测试_数量 = len(测试用数据载入器.dataset)

        for 轮回 in range(恢复的轮回数, 轮回数):
            for 阶段 in ['训练', '验证']:
                开始时间 = timeit.default_timer()

                运行时损失值 = 0.0
                运行时准确个数 = 0.0

                if 阶段 == '训练':
                    调度器.step()
                    模型.train()
                else:
                    模型.eval()

                for 复数输入, 复数标签 in tqdm(训练验证_载入器[阶段]):
                    复数输入 = Variable(复数输入, requires_grad=True).to(设备)
                    复数标签 = Variable(复数标签).to(设备)
                    优化器.zero_grad()

                    if 阶段 == '训练':
                        复数输出 = 模型(复数输入)
                    else:
                        with torch.no_grad():
                            复数输出 = 模型(复数输入)

                    概率 = nn.Softmax(dim=1)(复数输出)
                    预测 = torch.max(概率, 1)[1]
                    损失 = 准绳(复数输出, 复数标签.long())

                    if 阶段 == '训练':
                        损失.backward()
                        优化器.step()

                    运行时损失值 += 损失.item() * 复数输入.size(0)
                    运行时准确个数 += torch.sum(预测 == 复数标签.data)

                轮回损失值 = 运行时损失值 / 训练验证_数量[阶段]
                轮回准确率 = 运行时准确个数.double() / 训练验证_数量[阶段]

                if 阶段 == '训练':
                    作者.add_scalar('数据/训练时轮回损失值', 轮回损失值, 轮回)
                    作者.add_scalar('数据/训练时轮回准确率', 轮回准确率, 轮回)
                else:
                    作者.add_scalar('数据/验证时轮回损失值', 轮回损失值, 轮回)
                    作者.add_scalar('数据/验证时轮回准确率', 轮回准确率, 轮回)

                print("[{}] 轮回：{}/{} 损失值：{} 准确率：{}".format(阶段, 轮回 + 1, 轮回数, 轮回损失值, 轮回准确率))
                停止时间 = timeit.default_timer()
                print("执行时间：" + str(停止时间 - 开始时间) + '\n')

            if 轮回 % 保存轮回间隔 == (保存轮回间隔 - 1):
                torch.save({
                    '轮回': 轮回 + 1,
                    '状态字典': 模型.state_dict(),
                    '优化器字典': 优化器.state_dict()
                }, os.path.join(保存路径, '模型', 保存名 + '_轮回-' + str(轮回) + '.pth.tar'))
                print("在{}保存模型\n".format(os.path.join(保存路径, '模型', 保存名 + '_轮回-' + str(轮回) + '.pth.tar')))

            if 是否测试 and 轮回 % 测试间隔 == (测试间隔 - 1):
                模型.eval()
                开始时间 = timeit.default_timer()
                运行时损失值 = 0.0
                运行时准确率 = 0.0

                for 复数输入, 复数标签 in tqdm(测试用数据载入器):
                    复数输入 = Variable(复数输入, requires_grad=True).to(设备)
                    复数标签 = Variable(复数标签).to(设备)
                    优化器.zero_grad()
                    with torch.no_grad():
                        复数输出 = 模型(复数输入)

                    概率 = nn.Softmax(dim=1)(复数输出)
                    预测 = torch.max(概率, 1)[1]
                    损失 = 准绳(复数输出, 复数标签.long())

                    运行时损失值 += 损失.item() * 复数输入.size(0)
                    运行时准确率 += torch.sum(预测 == 复数标签.data)

                轮回损失值 = 运行时损失值 / 测试_数量
                轮回准确率 = 运行时准确率.double() / 测试_数量

                作者.add_scalar('数据/测试时轮回损失值', 轮回损失值, 轮回)
                作者.add_scalar('数据/测试时轮回准确率', 轮回准确率, 轮回)

                print("[测试] 轮回：{}/{} 损失值：{} 准确率：{}".format(轮回 + 1, 轮回数, 轮回损失值, 轮回准确率))
                停止时间 = timeit.default_timer()
                print("执行时间：" + str(停止时间 - 开始时间) + '\n')

        作者.close()

    训练模型()
