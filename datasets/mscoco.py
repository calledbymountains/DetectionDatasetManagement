from datasets.DatasetBase import DatasetBase
import os
import glob
import json


class MSCOCO(DatasetBase):
    def __init__(self, name, year, base_dir, save_dir, subset):
        super(MSCOCO, self).__init__(name, save_dir=save_dir)
        self._basedir = self.folderpresence(base_dir, create=False)
        if subset not in ['train', 'val', 'test']:
            raise (ValueError, 'Subset muse be either of "train", "val" or "test".')
        self._subset = subset
        self._year = year
        self._imagepath = self.folderpresence(os.path.join(self._basedir, '{}{}'.format(self._subset,
                                                                                        self._year)))
        self._annpath = self.folderpresence(os.path.join(self._basedir, 'annotations'))

    def imagefiles(self):
        files = glob.glob(os.path.join(self._imagepath, '*.jpg'))
        return files



