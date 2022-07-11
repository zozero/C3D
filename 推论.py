# inference.py 用于测试结果
import cv2
import numpy as np
import torch

from 网络 import 三维卷积模型类


def 裁剪中间(帧):
    帧 = 帧[8:120, 30:142, :]
    return np.array(帧).astype(np.uint8)


def 主要():
    设备 = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("%s设备将被使用" % 设备)

    with open('./数据载入器/ucf_标签列表.txt', 'r') as f:
        类名 = f.readlines()
        f.close()

    模型 = 三维卷积模型类.三维卷积(分类数=101)
    检查点 = torch.load('./运行/运行_1/模型/三维卷积-ucf101_轮回-24.pth.tar', map_location=lambda storage, loc: storage)

    模型.load_state_dict(检查点['状态字典'])

    模型.to(设备)
    模型.eval()

    视频 = './数据/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c06.avi'
    捕获 = cv2.VideoCapture(视频)
    是否保留 = True

    修剪 = []

    while 是否保留:
        是否保留, 帧 = 捕获.read()
        if not 是否保留 and 帧 is None:
            continue

        临时_ = 裁剪中间(cv2.resize(帧, (171, 128)))
        临时 = 临时_ - np.array([[[90.0, 98.0, 102.0]]])
        修剪.append(临时)
        if len(修剪) == 16:
            复数输入 = np.array(修剪).astype(np.float32)
            复数输入 = np.expand_dims(复数输入, axis=0)
            复数输入 = np.transpose(复数输入, (0, 4, 1, 2, 3))
            复数输入 = torch.from_numpy(复数输入)
            复数输入 = torch.autograd.Variable(复数输入, requires_grad=False).to(设备)
            with torch.no_grad():
                复数输出 = 模型.forward(复数输入)

            概率 = torch.nn.Softmax(dim=1)(复数输出)
            标签 = torch.max(概率, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(帧, 类名[标签].split(' ')[-1].strip(), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(帧, "prob:%.4f" % 概率[0][标签], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            修剪.pop(0)

        cv2.imshow('result', 帧)
        cv2.waitKey(30)

    捕获.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    主要()
