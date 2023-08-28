import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseModel
from module.backbone import BACKBONE
from models.cls_models.TextNet import TextNetFeature
from torch.nn.functional import normalize
import torch
from module.head import ResClsLessCNN
from module.non_local import NLBlockND

class ResNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(ResNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=pretrained)
        if backbone[:3] == 'vgg':
            scale = 16
        elif int(backbone[6:]) > 34:
            scale = 64
        elif int(backbone[6:]) <= 34:
            scale = 16
        else:
            raise Exception('Unknown backbone')
        self.cls_branch = ResClsLessCNN(scale=scale, num_class=num_classes)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, image):
        self.res = self.backbone(image)
        out = self.cls_branch(self.res[-1])
        return out

    def get_feats(self):
        feats = F.adaptive_avg_pool2d(self.res[-1], (1, 1))
        feats = feats.view(feats.size(0), -1)
        return feats

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out, attention

class ResNetFusionTextNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(ResNetFusionTextNet, self).__init__(backbone, n_channels, num_classes, pretrained)

        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=pretrained)
        self.text_net = TextNetFeature(backbone=backbone, n_channels=n_channels,
                                       num_classes=num_classes, pretrained=pretrained)
        self.feature_dim = 512
        self.filter_num = 32
        self.text_dim = self.text_net.hidden[-1]

        self.atten = Self_Attn(self.feature_dim + self.text_dim)

        # ResNet、VGG
        if backbone[:3] == 'vgg':
            self.scale1 = 16
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        elif int(backbone[6:]) > 34:
            self.scale1 = 64
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        elif int(backbone[6:]) <= 34:
            self.scale1 = 16
            self.conv1 = nn.Sequential(
                nn.Linear(self.filter_num * self.scale1, num_classes),
            )
        else:
            raise Exception('Unknown backbone')

        self.nlb4 = NLBlockND(in_channels=self.filter_num * self.scale1, mode='concatenate', dimension=2,
                              bn_layer=True)

        self.aw = nn.Parameter(torch.zeros(2))
        self.ca = nn.Parameter(torch.zeros(1))

        self.softmax2d = nn.Softmax2d()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fusion_projector = nn.Sequential(
            nn.Linear(self.feature_dim + self.text_dim, self.feature_dim + self.text_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim + self.text_dim, self.feature_dim // 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.text_net.hidden[-1] + self.feature_dim, self.text_net.hidden[-1]),
            nn.ReLU(),
            nn.Linear(self.text_dim, num_classes),

        )

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, x, text, text_included=False, cam=None):
        h = self.backbone(x)
        nlb = self.nlb4(h[-1])

        if text_included:
            text_w = self.aw[0]
            text_feature = self.text_net(text)  # 临床特征 shape = [32, 64, 1]
            text = self.text_net.num(text.permute(0, 2, 1))
            text_feature = text_feature + text_w * self.softmax2d(text_feature) * text  # + text
            text_feature = text_feature.unsqueeze(-1).expand(-1, 64, 7, 7)

            vis_feature = nlb
            x_fusion = torch.cat((vis_feature, text_feature), dim=1)  # (32, 512+64, 7, 7)

            out, attention = self.atten(x_fusion)
            out_i = self.avgpool(out)
            x_fusion = torch.flatten(out_i, 1)
            contrs_learn = x_fusion
            self.feats = x_fusion

            out = self.classifier(x_fusion)
        else:
            x = F.adaptive_avg_pool2d(h[-1], (1, 1))
            x = x.view(x.size(0), -1)
            out = self.conv1(x)
        z_i = normalize(self.fusion_projector(contrs_learn), dim=1)
        return out, z_i

    def get_feats(self):
        return self.feats

if __name__ == '__main__':
    rft = ResNetFusionTextNet('resnet18', 3, 2, True)
    print(rft)
