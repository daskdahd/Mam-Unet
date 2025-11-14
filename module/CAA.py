# # 使用前需要先安装mmcv，建议3.9版本python，
# # 安装方法：1.终端输入pip install -U openmim   2.终端输入mim install "mmcv>=2.0.0rc1" 安装完成
# from typing import Optional  # 引入Optional类型提示，用于表示某个参数可以是None。
# import torch  # 引入PyTorch库，用于深度学习的张量操作。
# import torch.nn as nn  # 引入PyTorch的神经网络模块，用于构建神经网络层。
# from mmcv.cnn import ConvModule  # 引入mmcv中的ConvModule模块，用于方便地创建卷积层。
# from mmengine.model import BaseModule  # 引入mmengine中的BaseModule基类，用于自定义模块。

# class CAA(BaseModule):  # 定义一个名为CAA的类，继承自BaseModule，用于实现“上下文锚点注意力”机制。
#     """Context Anchor Attention"""  # 简要描述该模块的作用。
#     def __init__(
#             self,
#             channels: int,  # 输入特征图的通道数。
#             h_kernel_size: int = 11,  # 水平方向卷积核的大小，默认为11。
#             v_kernel_size: int = 11,  # 垂直方向卷积核的大小，默认为11。
#             norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),  # 归一化配置，默认使用批归一化（BN）。
#             act_cfg: Optional[dict] = dict(type='SiLU'),  # 激活函数配置，默认使用SiLU激活函数。
#             init_cfg: Optional[dict] = None,  # 初始化配置，默认为None。
#     ):
#         super().__init__(init_cfg)  # 调用父类的初始化方法，传入初始化配置。
#         # 定义平均池化层，核大小为7，步幅为1，填充为3（保持特征图尺寸不变）。
#         self.avg_pool = nn.AvgPool2d(7, 1, 3)  # 通过平均池化获取局部区域特征
#         # 定义第一个卷积模块，1x1卷积，用于调整通道数和进行非线性变换。
#         self.conv1 = ConvModule(channels, channels, 1, 1, 0,  # 通过1*1卷积减少计算量
#                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
#         # 通过分别对水平和垂直方向的特征进行处理，网络能够更好地捕捉到不同方向的局部特征，提高模型对复杂场景中细节的捕捉能力
#         # 定义水平卷积模块，卷积核为(1, h_kernel_size)，按通道分组卷积，保持通道数不变。
#         self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
#                                  (0, h_kernel_size // 2), groups=channels,
#                                  norm_cfg=None, act_cfg=None)
#         # 定义垂直卷积模块，卷积核为(v_kernel_size, 1)，按通道分组卷积，保持通道数不变。
#         self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
#                                  (v_kernel_size // 2, 0), groups=channels,
#                                  norm_cfg=None, act_cfg=None)
#         # 定义第二个卷积模块，再次使用1x1卷积，用于进一步调整通道数和进行非线性变换。
#         self.conv2 = ConvModule(channels, channels, 1, 1, 0,
#                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
#         # 定义Sigmoid激活函数，用于生成注意力权重。
#         self.act = nn.Sigmoid()

#     def forward(self, x):  # 定义前向传播函数。
#         # 输入经过平均池化、卷积模块和激活函数，最终生成注意力因子。
#         attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
#         return x * attn_factor    # 将注意力因子与原始输入特征图逐元素相乘，调整特征图的权重，突出重要特征。
    
# # 测试CAA模块
# if __name__ == "__main__":  # 如果此脚本作为主程序运行，则执行以下代码。
#     x = torch.randn(4, 64, 32, 32)  # 生成一个随机张量作为输入，形状为(4, 64, 32, 32)。
#     caa = CAA(64)  # 创建CAA模块实例，输入通道数为64。
#     out = caa(x)  # 将输入张量通过CAA模块。
#     # print(out)
#     print(out.shape)  # 打印输出张量的形状，应该为torch.Size([4, 64, 32, 32])。


from typing import Optional
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return x  * attn_factor

# 测试CAA模块
if __name__ == "__main__":  # 如果此脚本作为主程序运行，则执行以下代码。
    x = torch.randn(4, 64, 32, 32)  # 生成一个随机张量作为输入，形状为(4, 64, 32, 32)。
    caa = CAA(64)  # 创建CAA模块实例，输入通道数为64。
    out = caa(x)  # 将输入张量通过CAA模块。
    # print(out)
    print(out.shape)  # 打印输出张量的形状，应该为torch.Size([4, 64, 32, 32])。