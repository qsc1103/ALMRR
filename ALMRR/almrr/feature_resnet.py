import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import resnet50, wide_resnet101_2

class MultiScaleResNet50(torch.nn.Module):
    def __init__(self):
        super(MultiScaleResNet50, self).__init__()
        # 加载预训练的ResNet50模型
        weights_path = r'./pretrained/resnet50-19c8e357.pth'
        state_dict = torch.load(weights_path)
        original_model = resnet50()
        original_model.load_state_dict(state_dict)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        # print(self.features)
        # 提取不同阶段的层
        self.layer1 = self.features[:5]
        # print(self.layer1)
        self.layer2 = self.features[5:6]
        # print(self.layer2)
        self.layer3 = self.features[6:7]
        # print(self.layer3)
        self.layer4 = self.features[7:]
        # print(self.layer4)
#
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


model = MultiScaleResNet50()
model.to('cuda:0')
model.eval()



# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, kernel_size, output_size=None, dilation=1, stride=1, device=torch.device('cpu')):
        super(AvgFeatAGG2d, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        output = self.unfold(input)  # (b, cxkxk, h*w)
        output = torch.reshape(output, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
        output = torch.mean(output, dim=2)
        return output


class Extractor_ResNet50(nn.Module):
    r"""
    Build muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(self, upsample="bilinear",
                 is_agg=True,
                 kernel_size=(4, 4),
                 stride=(4, 4),
                 dilation=1,
                 featmap_size=(256, 256),
                 device='cuda:0'):

        super(Extractor_ResNet50, self).__init__()
        self.device = torch.device(device)
        self.is_agg = is_agg
        self.map_size = featmap_size
        self.upsample = upsample
        self.patch_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # feature processing
        padding_h = (self.patch_size[0] - self.stride[0]) // 2
        padding_w = (self.patch_size[1] - self.stride[1]) // 2
        self.padding = (padding_h, padding_w)
        self.replicationpad = nn.ReplicationPad2d((padding_w, padding_w, padding_h, padding_h))
        self.out_h = int((self.map_size[0] + 2*self.padding[0] - (self.dilation * (self.patch_size[0] - 1) + 1)) / self.stride[0] + 1)
        self.out_w = int((self.map_size[1] + 2*self.padding[1] - (self.dilation * (self.patch_size[1] - 1) + 1)) / self.stride[1] + 1)
        self.out_size = (self.out_h, self.out_w)
        self.feat_agg = AvgFeatAGG2d(kernel_size=self.patch_size, output_size=self.out_size,
                                    dilation=self.dilation, stride=self.stride, device=self.device)
        self.unfold = nn.Unfold(kernel_size=1, dilation=1, padding=0, stride=1)

    def forward(self, input):
        feat_maps = model(input)
        feat_maps = {'first_block':feat_maps[0], 'second_block':feat_maps[1], 'third_block':feat_maps[2]}
        features = torch.Tensor().to(self.device)
        # extracting features
        for _, feat_map in feat_maps.items():
            if self.is_agg:
                feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
                feat_map = self.replicationpad(feat_map)
                feat_map = self.feat_agg(feat_map)
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
            else:
                feat_map = nn.functional.interpolate(feat_map, size=self.out_size, mode=self.upsample)
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
        b, c, n = features.shape
        features = torch.reshape(features, (b, c, self.out_size[0], self.out_size[1]))

        return features
