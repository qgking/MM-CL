import sys
sys.path.extend(["../../", "../", "./"])
from einops import rearrange
from os.path import isdir
from torch.utils.data import Dataset
from commons.constant import *
from commons.utils import *
import os

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

class SarcopeniaBasePTHDataSet(Dataset):
    def __init__(self, sarcopenia_paths_dfs, input_size=(256, 256), augment=False, text_only=False):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.sarcopeniaidx = []
        self.sarcopenialines = []
        self.augment = augment
        self.img_final_dfs = sarcopenia_paths_dfs
        self.text_only = text_only
        self.pos_dfs = []
        self.neg_dfs = []
        for dfs in sarcopenia_paths_dfs:
            cat = dfs[CAT]
            if cat == 0:
                self.neg_dfs.append(dfs)
            elif cat == 1:
                self.pos_dfs.append(dfs)
            else:
                raise Exception('Unknown CAT')

        print('Load images: %d,positive: %d, negative: %d' % (
            len(self.img_final_dfs), len(self.pos_dfs), len(self.neg_dfs)))

    def __len__(self):
        return int(len(self.img_final_dfs))

    def __getitem__(self, index):
        return

class SarcopeniaClsPTHDataSet(SarcopeniaBasePTHDataSet):
    def __init__(self, sarcopenia_paths_dfs, input_size, augment, text_only):
        super(SarcopeniaClsPTHDataSet, self).__init__(sarcopenia_paths_dfs, input_size, augment, text_only)

    def __getitem__(self, index):
        dfs = self.img_final_dfs[index]
        img = dfs[IMG]
        path = dfs[PATH]
        name = dfs[NAME]
        cat = dfs[CAT]
        nums = dfs[NUMS]
        image_patch = img
        if self.augment:
            image_patch = self.augment(np.transpose(img.astype(np.uint8), (1, 2, 0)))
        numerical_tensor = torch.from_numpy(rearrange(nums, 'b -> () () b'))
        return {
            "image_patch": image_patch,
            "image_cat": cat,
            "image_name": name,
            "image_path": path,
            "image_text": numerical_tensor
        }

if __name__ == '__main__':
    print()