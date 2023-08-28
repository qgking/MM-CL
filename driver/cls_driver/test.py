import matplotlib
import sys
sys.path.extend(["../../", "../", "./"])
import torch
import random
from driver.cls_driver.ClsHelper import ClsHelper
from driver.Config import Configurable
from models import MODELS
from copy import deepcopy
from commons.constant import *
matplotlib.use("Agg")
import configparser
from driver import transform_test
from module.torchcam.methods import SmoothGradCAMpp
config = configparser.ConfigParser()
import argparse
from os.path import join
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from commons.utils import sensitivity_score, specificity_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
import torchvision.models as resnets

def main(config, args, seed=111):
    criterion = {}
    cls_help = ClsHelper(criterion, config)
    cls_help.move_to_cuda()
    print("data name ", cls_help.config.data_name)
    print("data size ", cls_help.config.patch_x)
    print("Random dataset Seed: %d" % (seed))

    model_path = '../../data/'

    train_dataloader, vali_loader, test_image_dfs = cls_help.get_data_loader_pth(fold=0, seed=seed)

    resnet = resnets.resnet18(pretrained=True).eval().to(cls_help.device)
    cam_extractor = SmoothGradCAMpp(model=resnet, target_layer='layer4')

    model = MODELS['ResNetFusionTextNet'](backbone='resnet18', n_channels=cls_help.config.n_channels,
                                          num_classes=cls_help.config.classes, pretrained=True)

    load_file = join(model_path, "MM-CL.pt")
    state_dict = torch.load(load_file, map_location=('cuda:' + str(cls_help.device)))
    model.load_state_dict(state_dict)
    model.to(cls_help.device)
    del state_dict
    gt, pred = test_mm_cl(cls_help, model, resnet, cam_extractor, test_image_dfs)
    del model

    auc = roc_auc_score(gt, pred[:,1])
    print('\nroc_auc : %.4f'%(auc))

    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(gt, pred)
    print('acc : %.4f'%(acc))

    f1 = f1_score(gt, pred)
    print('f1score : %.4f' % (f1))

    sen = sensitivity_score(pred, gt)
    print('sensitivity : %.4f' % (sen))

    spc = specificity_score(pred, gt)
    print('specificity : %.4f' % (spc))

    pre = precision_score(gt, pred)
    print('pre : %.4f\n'%(pre))

    cls_help.log.flush()
    cls_help.summary_writer.close()

def test_mm_cl(cls_help, model, resnet, cam_extractor, test_image_dfs):
    model_clone = deepcopy(model)
    model_clone.eval()
    num = 0
    labels = []
    for dfs in test_image_dfs:
        image = dfs[IMG]
        image_label = [dfs[CAT]]
        image_text = dfs[NUMS]
        image_text = torch.from_numpy(rearrange(image_text, 'd -> () () d')).to(cls_help.device)
        image_patch = transform_test(np.transpose(image.astype(np.uint8), (1, 2, 0))).to(cls_help.device)
        image_patch = rearrange(image_patch, 'C H W -> () C H W')
        scores = resnet(image_patch)
        activation_map = cam_extractor(scores.squeeze(0).argmax().item(), scores)
        activation_image = F.interpolate(torch.stack(activation_map).view(-1, 1, 7, 7),
                                         size=image_patch.size()[2:], mode='bilinear', align_corners=True)
        input = (1+torch.sigmoid(activation_image))*image_patch

        with torch.no_grad():
            logits,_ = model_clone(input, text=image_text, text_included=True)
        probs = F.softmax(logits, dim=1)
        if num == 0:
            num = 1
            probs_ = probs
        else:
            probs_ = torch.cat((probs_, probs), dim=0)
        labels.append(image_label[0])
    probs_ = probs_.detach().cpu().numpy()
    labels_ = np.array(labels)
    return labels_, probs_

if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='./config/cls_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=False)
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default='0')
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default="MM-CL")
    argparser.add_argument('--ema-decay', help='ema decay', default="0.99")
    argparser.add_argument('--seed', help='random seed', default=111, type=int)
    argparser.add_argument('--model', help='model name', default="ResNetFusionTextNet")
    argparser.add_argument('--backbone', help='backbone name', default="resnet18")

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config, args, seed=args.seed)
