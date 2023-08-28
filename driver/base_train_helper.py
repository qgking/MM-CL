import sys
sys.path.extend(["../../", "../", "./"])
from datetime import datetime
import shutil
from os.path import isdir, join
from commons.utils import *
import torch
from sklearn.model_selection import KFold
from torch.cuda import empty_cache
from models import MODELS
from mscv import create_summary_writer

class BaseTrainHelper(object):
    def __init__(self, criterions, config):
        self.criterions = criterions
        self.config = config
        self.use_cuda = config.use_cuda
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.reset_model()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.init_params()

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)
        code_path = join(self.config.submission_dir, 'code')
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        print(os.getcwd())
        shutil.copytree('../../', code_path, ignore=shutil.ignore_patterns('.git', '__pycache__', '*log*', '*tmp*'))

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            for key in self.criterions.keys():
                print(key)
                self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_n_fold(self, image_paths, fold, seed=666):
        kf = KFold(n_splits=self.config.nfold, random_state=seed, shuffle=True)
        cur_fold = 0
        for train_index, test_index in kf.split(image_paths):
            if cur_fold == fold:
                return train_index, test_index
            cur_fold += 1

    def create_model(self):
        mm = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.n_channels,
                                       num_classes=self.config.classes, pretrained=True)
        return mm

    def create_meteacher(self):
        mtc = self.create_model()
        mtc.to(self.device)
        return mtc

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
            empty_cache()
        print("Creating models....")
        self.model = self.create_model()
        self.model.to(self.device)

    def count_parameters(self, net):
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def out_put_summary(self):
        self.summary_writer = create_summary_writer(self.config.tensorboard_dir)
        print('Model has param %.2fM' % (self.count_parameters(self.model) / 1000000.0))
