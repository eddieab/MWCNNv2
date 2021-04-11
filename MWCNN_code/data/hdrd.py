import glob
import os

from data import srdata


class HDRD(srdata.SRData):
    def __init__(self, args, train=True):
        self.glob_argument = os.path.join(args.dir_data, args.data_train) + '/*'
        super(HDRD, self).__init__(args, train)
        self.repeat = 1

    def _scan(self):
        print('scan in HDRD')
        list_hr = sorted(glob.glob(self.glob_argument))
        self.num = len(list_hr)
        self.num_samples = self.num
        print('number of training samples: ', self.num)
        return list_hr

    def _set_filesystem(self, dir_data):
        self.glob_argument = os.path.join(dir_data, self.args.data_test) + '/*'
        return