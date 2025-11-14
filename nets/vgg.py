import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        feat1 = self.features[  :4 ](x)
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    """
    根据给定的配置列表构建 VGG 网络的特征提取层。

    参数:
    cfg (list): 网络配置列表，包含卷积层的输出通道数和最大池化层标识 'M'。
    batch_norm (bool, 可选): 是否在卷积层后添加批量归一化层，默认为 False。
    in_channels (int, 可选): 输入图像的通道数，默认为 3。

    返回:
    nn.Sequential: 包含卷积层、批量归一化层（可选）、ReLU 激活函数和最大池化层的顺序模块。
    """
    # 初始化一个空列表，用于存储网络层
    layers = []
    # 遍历配置列表中的每个元素
    for v in cfg:
        # 若元素为 'M'，表示添加一个最大池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 若元素不为 'M'，创建一个 3x3 的卷积层，padding 为 1 以保持特征图尺寸不变
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # 若 batch_norm 为 True，添加卷积层、批量归一化层和 ReLU 激活函数
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                # 若 batch_norm 为 False，仅添加卷积层和 ReLU 激活函数
                layers += [conv2d, nn.ReLU(inplace=True)]
            # 更新输入通道数为当前卷积层的输出通道数
            in_channels = v
    # 将存储网络层的列表转换为 nn.Sequential 模块并返回
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 此注释表示输入图像尺寸和通道数在经过特征提取层时的变化过程
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    
    del model.avgpool
    del model.classifier
    return model
