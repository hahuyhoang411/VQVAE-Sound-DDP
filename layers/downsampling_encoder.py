import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        prev_channels = 1
        dilation_factor = 1
        total_scale = 1
        pad_left = 0
        self.skips = []
        for scale, ksz, dilate in layer_specs:
            if dilate:
                dilation_factor *= scale
                stride = 1
            else:
                stride = scale
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)

            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            print(f'pad += {total_scale} * {ksz-stride} * {dilation_factor}')
            self.skips.append(skip)
            if not dilate:
                total_scale *= scale
        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)

    def forward(self, samples):
        x = samples.unsqueeze(1)
        #print(f'sd[samples] {x.std()}')
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            scale, ksz, dilate = layer_spec

            x1 = conv_wide(x)
            #print(f'sd[conv.s] {x1.std()}')
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            #print(f'sd[act] {x2.std()}')
            x3 = conv_1x1(x2)
            #print(f'sd[conv.1] {x3.std()}')
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*scale].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            #print(f'sd[out] {x.std()}')
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        #print(f'sd[final] {x.std()}')
        return x.transpose(1, 2)