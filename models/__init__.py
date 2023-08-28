import numpy as np

np.random.seed(0)
from models.cls_models.resnet import ResNet, ResNetFusionTextNet

MODELS = {
    'resnet34': ResNet,
    'resnet18': ResNet,
    'resnet50':ResNet,
    'resnet101': ResNet,
    'ResNetFusionTextNet': ResNetFusionTextNet,
}
