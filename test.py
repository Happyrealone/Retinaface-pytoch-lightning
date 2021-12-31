import timm
import torch
import torchvision.models
import torchvision.models._utils as _utils
from torchvision.models.feature_extraction import create_feature_extractor

import tensorboardX
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from pprint import pprint
from torchvision.models.mobilenetv3 import mobilenet_v3_small

# 查看timm已支持的backobone
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


# 初始化timm模型并加载权重
# model = timm.create_model("efficientnet_es", pretrained=True, features_only=True)
# feature_out = model(torch.rand(1, 3, 840, 840))
# for feature in feature_out:
#     print(feature.shape)

# model = mobilenet_v3_small(pretrained=True)
# print(model)


# 采用fx方式提取特征图
# feature_extractor = create_feature_extractor(model, return_nodes=['blocks.2', 'blocks.4', 'blocks.6'])
# output = feature_extractor(torch.rand(1, 3, 840, 840))
# input = []
# for key, feature in output.items():
#     input.append(feature)


# torchvision中anchor的生成
# image_list = ImageList(torch.zeros(size=(1,3,840,840)), [([840,840])])
# get_anchors = AnchorGenerator(sizes=((16, 32), (64, 128), (256, 512)), aspect_ratios=((1),))
# anchors = get_anchors(image_list, feature_out[-3:])


