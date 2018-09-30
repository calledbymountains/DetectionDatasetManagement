import abc
import os
import pandas as pd



class DatasetBase(metaclass=abc.ABCMeta):
    def __init__(self, name, save_dir):
        self._name = name
        self._savedir = self.folderpresence(save_dir, create=True)

    @property
    def name(self):
        return self._name

    @property
    def savedir(self):
        return self._savedir

    @abc.abstractmethod
    def annotations(self):
        return

    def writedataframe(self, overwrite=False):
        savename = '{}.csv'.format(os.path.join(self.savedir, self.name))
        if os.path.exists(savename):
            if not overwrite:
                return None
        annotations = self.annotations()
        df = pd.DataFrame(annotations)
        df.to_csv(savename)
        return None

    def folderpresence(self, folder, create=False):
        if not os.path.isdir(folder):
            if create:
                try:
                    os.makedirs(folder)
                    return folder
                except OSError:
                    raise (OSError, 'The folder {} does not appear to be a valid folder.'.format(folder))
            else:
                print(folder)
                raise (OSError, 'The folder {} does not exist.'.format(folder))
        else:
            return folder
