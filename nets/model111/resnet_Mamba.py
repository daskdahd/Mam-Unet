import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append(r"D:\yyb\project\unet-pytorch-main")
from module.ECA import *
from module.Biformer import BiLevelRoutingAttention as BRA
from module.transMamba import TransMambaBlock  # 新增：导入TransMamba模块

# 定义一个3x3卷积层的辅助函数
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    创建3x3卷积层的工厂函数
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数  
        stride: 步长，默认为1
        groups: 分组卷积的组数，默认为1（标准卷积）
        dilation: 膨胀率，默认为1（标准卷积）
    Returns:
        nn.Conv2d: 配置好的3x3卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 定义一个1x1卷积层的辅助函数
def conv1x1(in_planes, out_planes, stride=1):
    """
    创建1x1卷积层的工厂函数，主要用于通道维度变换
    Args:
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride: 步长，默认为1
    Returns:
        nn.Conv2d: 配置好的1x1卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 基本残差块的定义（用于ResNet18/34）
class BasicBlock(nn.Module):
    expansion = 1  # 通道扩展倍数，BasicBlock不扩展通道数

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        BasicBlock初始化
        Args:
            inplanes: 输入通道数
            planes: 主要卷积层的通道数
            stride: 第一个卷积的步长
            downsample: 下采样层（当维度不匹配时使用）
            groups: 分组卷积参数
            base_width: 基础宽度参数
            dilation: 膨胀卷积参数
            norm_layer: 归一化层类型
        """
        super(BasicBlock, self).__init__()
        
        # 默认使用BatchNorm2d作为归一化层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # BasicBlock只支持标准配置
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            
        # 第一个3x3卷积：可能改变空间尺寸
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个3x3卷积：保持空间尺寸不变
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        # 残差连接的下采样层
        self.downsample = downsample
        self.stride = stride
        
        # 预留的注意力机制接口（目前被注释）
        #self.eca = ECA_layer(planes * self.expansion)  # ECA注意力
        # self.ema = EMA(channels=out_channels)          # EMA注意力
        # self.ela = ELA(out_channels,phi="T")           # ELA注意力
        # self.bra = BRA(out_channels)                   # BiFormer注意力

    def forward(self, x):
        """
        BasicBlock前向传播
        实现残差连接：output = F(x) + x
        """
        identity = x  # 保存输入作为残差连接

        # 第一个卷积块：Conv → BN → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块：Conv → BN
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入输出维度不匹配，对identity进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：F(x) + x
        out += identity
        out = self.relu(out)  # 最后的ReLU激活
        
        # 可选的注意力机制（目前被注释）
        #out = self.eca(out)

        return out

# Bottleneck残差块的定义（用于ResNet50及以上）
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩展倍数，输出通道是planes的4倍

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        Bottleneck残差块初始化
        采用1x1 → 3x3 → 1x1的瓶颈结构，减少计算量
        """
        super(Bottleneck, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # 计算中间层的通道宽度
        width = int(planes * (base_width / 64.)) * groups
        
        # 第一个1x1卷积：降维，减少通道数
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        # 第二个3x3卷积：特征提取，可能改变空间尺寸
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        # 第三个1x1卷积：升维，恢复通道数并扩展
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Bottleneck前向传播
        实现瓶颈残差连接：1x1降维 → 3x3特征提取 → 1x1升维 + 残差连接
        """
        identity = x  # 保存输入作为残差连接

        # 第一个1x1卷积块：降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个3x3卷积块：特征提取
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 第三个1x1卷积块：升维
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果输入输出维度不匹配，对identity进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：F(x) + x
        out += identity
        out = self.relu(out)

        return out

# ResNet主干网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, use_transmamba=True):  # 修改：添加use_transmamba参数
        """
        ResNet网络初始化
        Args:
            block: 残差块类型（BasicBlock或Bottleneck）
            layers: 每个stage的block数量列表，如[3,4,6,3]
            num_classes: 分类数量（本实现中不使用，因为是特征提取器）
            use_transmamba: 是否在layer4后添加TransMamba模块  # 新增参数
            
        网络结构（以600x600输入为例）：
        - conv1: 600x600x3 → 300x300x64
        - maxpool: 300x300x64 → 150x150x64  
        - layer1: 150x150x64 → 150x150x256
        - layer2: 150x150x256 → 75x75x512
        - layer3: 75x75x512 → 38x38x1024
        - layer4: 38x38x1024 → 19x19x2048
        - transmamba: 19x19x2048 → 19x19x2048 (增强特征)
        """
        self.inplanes = 64  # 当前层的输入通道数，动态更新
        super(ResNet, self).__init__()
        
        # Stem部分：初始特征提取
        # 输入图像 → 下采样2倍 + 通道扩展
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化：再次下采样2倍
        # 注意：ceil_mode=True确保输出尺寸向上取整
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        
        # 四个残差stage，逐步下采样并增加通道数
        self.layer1 = self._make_layer(block, 64, layers[0])          # 不下采样
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 下采样2倍
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 下采样2倍  
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 下采样2倍
        
        # 新增：添加TransMamba到layer4之后
        self.use_transmamba = use_transmamba 
        if use_transmamba:
            # layer4输出通道数为512 * expansion (对于Bottleneck是2048)
            layer4_channels = 512 * block.expansion
            self.transmamba = TransMambaBlock(
                dim=layer4_channels,
                num_heads=8,  # 可调节的注意力头数
                ffn_expansion_factor=1.5,  # FFN扩展因子
                bias=False,
                LayerNorm_type='BiasFree'
            )
        
        # 分类层（在分割任务中会被删除）
        self.avgpool = nn.AvgPool2d(7)  # 全局平均池化
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化：使用He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：He正态分布初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：权重初始化为1，偏置初始化为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建每个stage的残差层
        Args:
            block: 残差块类型
            planes: 该stage的基础通道数
            blocks: 该stage包含的block数量
            stride: 该stage的步长（第一个block使用）
        Returns:
            nn.Sequential: 包含多个残差块的序列
        """
        downsample = None
        
        # 判断是否需要下采样：
        # 1. 步长不为1（空间尺寸变化）
        # 2. 输入输出通道数不匹配
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个block：可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # 更新输入通道数
        self.inplanes = planes * block.expansion
        
        # 后续block：只进行特征提取，不改变尺寸
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播：返回多尺度特征用于分割任务
        在layer4后添加TransMamba增强高级语义特征
        
        与标准ResNet不同，这里返回多个中间特征图而不是分类结果
        这些特征图将用于U-Net的跳跃连接
        """
        
        # 原始ResNet分类流程（已注释）：
        # x = self.conv1(x)      # Stem卷积
        # x = self.bn1(x)        # Stem归一化
        # x = self.relu(x)       # Stem激活
        # x = self.maxpool(x)    # Stem池化
        # x = self.layer1(x)     # Stage1
        # x = self.layer2(x)     # Stage2  
        # x = self.layer3(x)     # Stage3
        # x = self.layer4(x)     # Stage4
        # x = self.avgpool(x)    # 全局池化
        # x = x.view(x.size(0), -1)  # 展平
        # x = self.fc(x)         # 分类

        # 分割任务的多尺度特征提取：
        # Stem部分
        x = self.conv1(x)        # 初始卷积：通道3→64，尺寸减半
        x = self.bn1(x)          # 归一化
        feat1 = self.relu(x)     # 第一层特征：高分辨率，低语义

        # 残差stage
        x = self.maxpool(feat1)  # 最大池化：尺寸再减半
        feat2 = self.layer1(x)   # 第二层特征：中等分辨率

        feat3 = self.layer2(feat2)  # 第三层特征：中等分辨率，中等语义
        feat4 = self.layer3(feat3)  # 第四层特征：低分辨率，高语义  
        feat5 = self.layer4(feat4)  # 第五层特征：最低分辨率，最高语义
        
        # 新增：在layer4后应用TransMamba增强特征
        if self.use_transmamba:
            feat5 = self.transmamba(feat5)  # 增强的第五层特征
        
        # 返回多尺度特征金字塔
        # 分辨率递减，语义信息递增
        return [feat1, feat2, feat3, feat4, feat5]

# ResNet50构建函数
def resnet50(pretrained=False, use_transmamba=True, **kwargs):  # 修改：新增use_transmamba参数
    """
    构建ResNet50骨干网络
    Args:
        pretrained: 是否加载ImageNet预训练权重
        use_transmamba: 是否使用TransMamba增强  # 新增参数说明
        **kwargs: 其他参数
    Returns:
        ResNet: 配置好的ResNet50特征提取器
    """
    # 创建ResNet50：Bottleneck + [3,4,6,3]配置
    # 总层数：3+4+6+3 = 16个Bottleneck × 3层卷积 + 2层额外 = 50层
    model = ResNet(Bottleneck, [3, 4, 6, 3], use_transmamba=use_transmamba, **kwargs)  # 修改：传递use_transmamba参数
    
    if pretrained:
        # 修改：更安全的预训练权重加载
        try:
            pretrained_dict = model_zoo.load_url(
                'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', 
                model_dir='model_data'
            )
            model_dict = model.state_dict()
            
            # 过滤掉TransMamba相关的权重（因为预训练模型中没有）
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                              if k in model_dict and 'transmamba' not in k}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print("成功加载预训练权重（TransMamba部分随机初始化）")
        except Exception as e:
            print(f"预训练权重加载失败: {e}")
            print("使用随机初始化权重")
    
    # 删除分类相关层，只保留特征提取部分
    # 这样网络就变成了纯特征提取器
    del model.avgpool  # 删除全局平均池化
    del model.fc       # 删除全连接分类层
    
    return model