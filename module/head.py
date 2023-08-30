import torch.nn as nn
import torch.nn.functional as F
from module.init_weights import weights_init_normal

class ResClsLessCNN(nn.Module):
    def __init__(self, filter_num=32, scale=16, num_class=2):
        super(ResClsLessCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Linear(filter_num * scale, num_class),
        )
        self.apply(weights_init_normal)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.conv1(x)
        return x
