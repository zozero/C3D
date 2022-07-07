class 路径:
    @staticmethod
    def 数据库目录(数据基目录):
        if 数据基目录 == 'ufc101':
            根目录 = "D:\\人工智能\\学习\\深度学习框架-PyTorch实战\\3维卷积\\数据\\UCF-101"
            输出目录 = "D:\\人工智能\\学习\\深度学习框架-PyTorch实战\\3维卷积\\已处理数据\\ufc101"
            return 根目录, 输出目录
        elif 数据基目录 == 'hmdb51':
            根目录 = "D:\\人工智能\\学习\\深度学习框架-PyTorch实战\\3维卷积\\数据\\hmdb-51"
            输出目录 = "D:\\人工智能\\学习\\深度学习框架-PyTorch实战\\3维卷积\\已处理数据\\hmdb51"
            return 根目录, 输出目录
        else:
            raise NotImplementedError("数据基目录{}不可用".format(数据基目录))

    @staticmethod
    def 模型目录():
        return './模型/c3d-pretrained.pth'
