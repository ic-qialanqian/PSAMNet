import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionLayer, self).__init__()
        # define the params
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups
        self.stride = stride
        # define the query key and value
        self.key_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        # define the relative position embedding
        # rel_col, rel_row = self._generate_rel_pos()
        # make it an parameters - we fix it and don't need the gradients
        # self.rel_col = nn.Parameter(torch.tensor(rel_col, dtype=torch.float32).unsqueeze(0), requires_grad=False)
        # self.rel_row = nn.Parameter(torch.tensor(rel_row, dtype=torch.float32).unsqueeze(0), requires_grad=False)

    def _generate_rel_pos(self):
        # use mesh grid to generate the relative position matrix
        rel_col, rel_row = np.meshgrid(np.arange(self.kernel_size), np.arange(self.kernel_size))
        rel_col = rel_col - (self.kernel_size - 1) * 0.5
        rel_row = rel_row - (self.kernel_size - 1) * 0.5
        # repeat
        rel_col = np.repeat(np.expand_dims(rel_col, 2), int(self.out_planes / 2), axis=2)
        rel_row = np.repeat(np.expand_dims(rel_row, 2), int(self.out_planes / 2), axis=2)
        # transpose the maps
        rel_col = np.transpose(rel_col, (2, 0, 1))
        rel_row = np.transpose(rel_row, (2, 0, 1))
        return rel_col, rel_row

    def forward(self, x):
        batch, channels, height, width = x.size()
        # padding the inputs - but not for the feature map send into the q
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        # generate the query, keys and the value
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)
        # start the next steps - unfold the output into windwos for easy sliding window multiplication
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # split the v for relative position
        #v_out_col, v_out_row = v_out.split(int(self.out_planes / 2), dim=1)
        #v_out = torch.cat((v_out_col + self.rel_col, v_out_row + self.rel_row), dim=1)
        #
        k_out = k_out.contiguous().view(batch, self.groups, int(self.out_planes / self.groups), height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, int(self.out_planes / self.groups), height, width, -1)
        q_out = q_out.view(batch, self.groups, int(self.out_planes / self.groups), height, width, 1)
        # sum
        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        # the final output
        return out

if __name__ == '__main__':
    attention_layer = AttentionLayer(in_planes=5, out_planes=20, kernel_size=5)
