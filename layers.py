# wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal_, kaiming_normal_
import copy
from functools import partial


# 初始化权重 对xavier_normal_和kaiming_normal_进行了封装，根据激活函数不同采用不同初始化
# xavier_normal_ 适用于线性激活函数
# kaiming_normal_ 用于非线性
def get_weight_init_fn(activation_fn):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """

    fn = activation_fn
    if hasattr(activation_fn, 'func'):
        fn = activation_fn.func
    # 如果采用LeakyReLU，则采用kaiming_normal_ 其a采用negative_slope
    # a -这层之后使用的rectifier的斜率系数 仅用于LeakyReLU
    if fn == nn.LeakyReLU:
        negative_slope = 0
        if hasattr(activation_fn, 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr(activation_fn, 'args'):
            if len(activation_fn.args) > 0:
                negative_slope = activation_fn.args[0]
        return partial(kaiming_normal_, a=negative_slope)
    # 采用ReLU或者PReLU，则采用kaiming_normal_ 并且a =0
    elif fn == nn.ReLU or fn == nn.PReLU:
        return partial(kaiming_normal_, a=0)
    # 其他的激活函数都采用xavier_normal_
    else:
        return xavier_normal_
    return


# 卷积层
def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None, use_batchnorm=False,
         pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias
    # 卷积层集合
    layers = []
    # 如果在卷积前激活，那么在层集合中添加预处理函数层
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    # 初始化卷积层
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    # 如果没初始化权重那么要初始化权重
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    try:
        weight_init_fn(conv.weight)
    except:
        print(conv.weight)
    # 将卷积层添加到层中
    layers.append(conv)
    # 如果是卷积后激活，那么在层集合中添加处理函数层
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    # 序列化
    return nn.Sequential(*layers)


# 反卷积(代码顺序同卷积，只是将卷积模块换成了反卷积)
def deconv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, activation_fn=None,
           use_batchnorm=False, pre_activation=False, bias=True, weight_init_fn=None):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(deconv.weight)
    layers.append(deconv)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


# 全连接层（同卷积层，只是将卷积换成Linear）
def linear(in_channels, out_channels, activation_fn=None, use_batchnorm=False, pre_activation=False, bias=True,
           weight_init_fn=None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(in_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    linear = nn.Linear(in_channels, out_channels)
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn(activation_fn)
    weight_init_fn(linear.weight)

    layers.append(linear)
    if not pre_activation:
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation_fn is not None:
            layers.append(activation_fn())
    return nn.Sequential(*layers)


# 基本卷积块（此卷积块也就是残差块）
class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_batchnorm=False,
                 activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=partial(nn.ReLU, inplace=True),
                 pre_activation=False, scaling_factor=1.0):
        super(BasicBlock, self).__init__()
        # 定义第一次卷积
        self.conv1 = conv(in_channels, out_channels, kernel_size, stride, kernel_size // 2, activation_fn,
                          use_batchnorm)
        # 定义第二次卷积
        self.conv2 = conv(out_channels, out_channels, kernel_size, 1, kernel_size // 2, None, use_batchnorm,
                          weight_init_fn=get_weight_init_fn(last_activation_fn))
        self.downsample = None
        # 如果输入通道数不等于输出通道数那么需要进行通道处理
        if stride != 1 or in_channels != out_channels:
            self.downsample = conv(in_channels, out_channels, 1, stride, 0, None, use_batchnorm)
        # 定义激活函数
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        # 残差因子
        self.scaling_factor = scaling_factor

    # 前向传播
    def forward(self, x):
        # 残差边
        residual = x
        # 是否下采样
        if self.downsample is not None:
            residual = self.downsample(residual)
        # 第一次卷积
        out = self.conv1(x)
        # 第二次卷积
        out = self.conv2(out)
        # 追加残差边
        out += residual * self.scaling_factor
        # 如果有激活函数，那么进行激活
        if self.last_activation is not None:
            out = self.last_activation(out)

        return out
