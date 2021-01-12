import torch
import torch.nn as nn
from layers import *
from conv_lstm import CLSTM_cell


# 5X5的卷积 激活函数采用RELU
def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


# 5X5的反卷积 激活函数采用RELU
def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))


# 残差块
def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


# EBlock块
class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(type(self), self).__init__()
        # 卷积
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        #  连续三个残差块
        for i in range(3):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    # 正向传播
    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


# DBlock块
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        # 连续三个残差块
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        # 反卷积层
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    # 正向传播
    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


# OutBlock块 输出
class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        # 连续三个残差
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        # 卷积 out_channels = 3 kernel_size = 5 stride = 1  padding = 2
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    # 正向传播
    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


# SRN网络模型
class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet 
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__(self, upsample_fn=partial(torch.nn.functional.upsample, mode='bilinear'), xavier_init_all=True):

        super(type(self), self).__init__()
        # 上采样
        self.upsample_fn = upsample_fn
        # 输入层 in_channels = 6 out_channels = 32 stride =1
        self.inblock = EBlock(3 + 3, 32, 1)
        # in_channels = 32 out_channels = 64 stride =2
        self.eblock1 = EBlock(32, 64, 2)
        # in_channels = 64 out_channels = 128 stride =2
        self.eblock2 = EBlock(64, 128, 2)
        # convlstm
        self.convlstm = CLSTM_cell(128, 128, 5)
        # in_channels = 128 out_channels=64 stride=2 output_padding=1
        self.dblock1 = DBlock(128, 64, 2, 1)
        # in_channels = 64 out_channels=32 stride=2 output_padding=1
        self.dblock2 = DBlock(64, 32, 2, 1)
        # in_channels = 32
        self.outblock = OutBlock(32)

        self.input_padding = None
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    # print(name)

    # 正向传递一次
    def forward_step(self, x, hidden_state):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h, c = self.convlstm(e128, hidden_state)
        d64 = self.dblock1(h)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        return d3, h, c

    # 正向传播 b1 b2 b3 分别是三个尺度 256，128，64
    def forward(self, b1, b2, b3):
        #
        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = torch.zeros_like(b3)
        # h，c为LSTM所需参数
        h, c = self.convlstm.init_hidden(b3.shape[0], (b3.shape[-2] // 4, b3.shape[-1] // 4))
        # self.input_padding 是全为0的矩阵，此次传播将b3 最小的尺度与 0拼接在一起，正向传播第一次
        i3, h, c = self.forward_step(torch.cat([b3, self.input_padding], 1), (h, c))
        # h，c上采 scale_factor = 2 代表输出是输入的2倍
        c = self.upsample_fn(c, scale_factor=2)
        h = self.upsample_fn(h, scale_factor=2)
        # 此处将i3的上采样和b2拼接然后，正向传播第二次
        i2, h, c = self.forward_step(torch.cat([b2, self.upsample_fn(i3, scale_factor=2)], 1), (h, c))
        # h,c上采
        c = self.upsample_fn(c, scale_factor=2)
        h = self.upsample_fn(h, scale_factor=2)
        # 此处将i2的上采样和b1拼接，最后一次传播
        i1, h, c = self.forward_step(torch.cat([b1, self.upsample_fn(i2, scale_factor=2)], 1), (h, c))

        # y2 = self.upsample_fn( y1 , (128,128) )
        # y3 = self.upsample_fn( y2 , (64,64) )
        # 返回三个尺度的输出
        # return y1 , y2 , y3
        return i1, i2, i3
