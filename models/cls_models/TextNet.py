from models.BaseModel import BaseModel
import torch
import torch.nn as nn
from sarcopenia_data.SarcopeniaDataLoader import TEXT_COLS

class TextNetFeature(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(TextNetFeature, self).__init__(backbone, n_channels, num_classes, pretrained)
        in_planes2 = len(TEXT_COLS)
        self.hidden = [64]
        self.num = nn.Conv1d(in_planes2, self.hidden[-1], kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.hidden[-1])
        self.silu2 = nn.SiLU(inplace=True)
        self.fc = nn.Linear(self.hidden[-1], self.hidden[-1])
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, num_x2):
        num_x = self.num(num_x2.permute(0, 2, 1))
        num_x = self.silu2(self.bn2(num_x))
        num_x = self.fc(num_x.squeeze(-1)).unsqueeze(-1)
        return num_x

if __name__ == '__main__':
    textnet = TextNetFeature('none', 3, 2)
    print(textnet)