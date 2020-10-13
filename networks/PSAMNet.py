import torch

from torch import nn

from torch.nn import init

import torch.nn.functional as F

import math

from torch.autograd import Variable

import numpy as np

from .deeplab_resnet import resnet50_locate

from networks.attention import AttentionLayer







config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'sum': [[512, 512, 256, 256], [512, 256, 256, 128], [False, True, True, True], [True, True, True, True]], 'score': 128}





class ChannelSELayer(nn.Module):

    """

    Re-implementation of Squeeze-and-Excitation (SE) block described in:

        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """



    def __init__(self, num_channels, reduction_ratio=2):

        """

        :param num_channels: No of input channels

        :param reduction_ratio: By how much should the num_channels should be reduced

        """

        super(ChannelSELayer, self).__init__()

        num_channels_reduced = num_channels // reduction_ratio

        self.reduction_ratio = reduction_ratio

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)

        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()



    def forward(self, input_tensor):

        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)

        :return: output tensor

        """

        batch_size, num_channels, H, W = input_tensor.size()

        # Average along each channel

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)



        # channel excitation

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))

        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))



        a, b = squeeze_tensor.size()

        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))

        return output_tensor









class ConvertLayer(nn.Module):

    def __init__(self, list_k):

        super(ConvertLayer, self).__init__()

        up = []

        for i in range(len(list_k[0])):

            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))

        self.convert0 = nn.ModuleList(up)



    def forward(self, list_x):

        resl = []

        for i in range(len(list_x)):

            resl.append(self.convert0[i](list_x[i]))

        return resl



class SumLayer(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):

        super(SumLayer, self).__init__()

        

        self.need_x2 = need_x2

        self.need_fuse = need_fuse

       

        convs = nn.Conv2d(k, k, 1, 1, 0, bias=False)

        self.convs = convs

        self.relu = nn.ReLU()

        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)

        if self.need_fuse:

            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)



    def forward(self, x, x2=None, x3=None):

        x_size = x.size()

        

        resl = self.convs(x)
        #resl = x

       

        resl = self.relu(resl)

        if self.need_x2:

            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)

        resl = self.conv_sum(resl)

        if self.need_fuse:

            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))

        return resl



class ScoreLayer(nn.Module):

    def __init__(self, k):

        super(ScoreLayer, self).__init__()

        self.score = nn.Conv2d(k ,1, 1, 1)



    def forward(self, x, x_size=None):

        x = self.score(x)

        if x_size is not None:

            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)

        return x



def extra_layer(base_model_cfg, vgg):

    if base_model_cfg == 'vgg':

        config = config_vgg

    elif base_model_cfg == 'resnet':

        config = config_resnet

    convert_layers, sum_layers, score_layers = [], [], []

    convert_layers = ConvertLayer(config['convert'])



    for i in range(len(config['sum'][0])):

        sum_layers += [SumLayer(config['sum'][0][i], config['sum'][1][i], config['sum'][2][i], config['sum'][3][i])]



    score_layers = ScoreLayer(config['score'])



    return vgg, convert_layers, sum_layers, score_layers





class PSAMNet(nn.Module):

    def __init__(self, base_model_cfg, base, convert_layers, sum_layers, score_layers):

        super(PSAMNet, self).__init__()

        self.base_model_cfg = base_model_cfg

        self.base = base

        self.sum = nn.ModuleList(sum_layers)

        self.score = score_layers

        
        self.ChannelSELayer1 = ChannelSELayer(512, 4)
        self.ChannelSELayer2 = ChannelSELayer(512, 4)
        self.ChannelSELayer3 = ChannelSELayer(256, 2)
        self.ChannelSELayer4 = ChannelSELayer(256, 2)

        

        
        if self.base_model_cfg == 'resnet':

            self.convert = convert_layers



    def forward(self, x):

        x_size = x.size()

        conv2merge, attentions = self.base(x)
        

        if self.base_model_cfg == 'resnet':

            conv2merge = self.convert(conv2merge)

        conv2merge = conv2merge[::-1]



        merge1 = self.sum[0](self.ChannelSELayer1(conv2merge[0]), self.ChannelSELayer2(conv2merge[1]),attentions[0])

        
        merge2 = self.sum[1](merge1, self.ChannelSELayer3(conv2merge[2]), attentions[1])

    
        merge3 = self.sum[2](merge2, self.ChannelSELayer4(conv2merge[3]), attentions[2])
       

        merge4 = self.sum[3](merge3, conv2merge[4],attentions[3])

               
        merge = self.score(merge4, x_size)

        return merge



def build_model(base_model_cfg='vgg'):

    if base_model_cfg == 'vgg':

        return PSAMNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))

    elif base_model_cfg == 'resnet':

        return PSAMNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))



def weights_init(m):

    if isinstance(m, nn.Conv2d):

        m.weight.data.normal_(0, 0.01)

        if m.bias is not None:

            m.bias.data.zero_()
