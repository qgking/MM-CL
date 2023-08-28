import sys
sys.path.extend(["../../", "../", "./"])
import torch
from driver.base_train_helper import BaseTrainHelper
from torch.utils.data import DataLoader
from sarcopenia_data.SarcopeniaDataLoader import SarcopeniaClsPTHDataSet
from driver import transform_local, transform_test
from os.path import join

class ClsHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(ClsHelper, self).__init__(criterions, config)

    def init_params(self):
        return

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_cat = [inst["image_cat"] for inst in batch]
        image_cat = torch.tensor(image_cat)
        image_name = [inst["image_name"] for inst in batch]
        image_path = [inst["image_path"] for inst in batch]
        image_text = [torch.unsqueeze(text, dim=0) for inst in batch for text in inst["image_text"]]
        image_text = torch.cat(image_text, dim=0)
        return {"image_patch": image_patch,
                "image_path": image_path,
                "image_name": image_name,
                "image_cat": image_cat,
                "image_text": image_text
                }

    def get_data_loader_pth(self, fold, seed=666, text_only=False):

        image_dfs = torch.load(join(self.config.data_path, 'sarcopenia_all_data.pth'))

        train_index, test_index = self.get_n_fold(image_paths=image_dfs, fold=fold, seed=seed)

        train_image_dfs = [image_dfs[index] for index in range(len(image_dfs)) if index in train_index]
        test_image_dfs = [image_dfs[index] for index in range(len(image_dfs)) if index in test_index]

        print("Train images %d: " % (len(train_image_dfs)))
        print("Test images %d: " % (len(test_image_dfs)))

        train_dataset = SarcopeniaClsPTHDataSet(train_image_dfs,
                                                input_size=(self.config.patch_x, self.config.patch_y),
                                                augment=transform_local, text_only=text_only)
        valid_dataset = SarcopeniaClsPTHDataSet(test_image_dfs,
                                                input_size=(self.config.patch_x, self.config.patch_y),
                                                augment=transform_test, text_only=text_only)

        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch,
                                  drop_last=True if len(train_image_dfs) % self.config.train_batch_size == 1 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch)

        return train_loader, valid_loader, test_image_dfs