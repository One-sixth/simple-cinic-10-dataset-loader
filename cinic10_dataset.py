import numpy as np
import os
from glob import glob
import threading
from multiprocessing import cpu_count
import imageio

class cinic10_dataset:
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def load(self, dataset_path='datasets/CINIC-10', in_memory=True):
        '''
        load from dataset dir
        :param dataset_path:
        :param in_memory: true if you want to load all image in memory.
                          false only load images path.
        '''
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.test_x = []
        self.test_y = []

        for idx, cls in enumerate(self.classes):
            xs = glob(os.path.join(dataset_path, 'train', cls, '*'))
            ys = [idx] * len(xs)
            self.train_x.extend(xs)
            self.train_y.extend(ys)
            xs = glob(os.path.join(dataset_path, 'valid', cls, '*'))
            ys = [idx] * len(xs)
            self.valid_x.extend(xs)
            self.valid_y.extend(ys)
            xs = glob(os.path.join(dataset_path, 'test', cls, '*'))
            ys = [idx] * len(xs)
            self.test_x.extend(xs)
            self.test_y.extend(ys)

        if in_memory:
            self.train_x = self._readimages(self.train_x)
            self.valid_x = self._readimages(self.valid_x)
            self.test_x = self._readimages(self.test_x)

        self.train_x = np.array(self.train_x)
        self.train_y = np.array(self.train_y)
        self.valid_x = np.array(self.valid_x)
        self.valid_y = np.array(self.valid_y)
        self.test_x = np.array(self.test_x)
        self.test_y = np.array(self.test_y)

    def load_from_npz(self, npz_file='cinic10.npz'):
        '''
        load dataset from npz. more fast than load from dir
        :param npz_file:
        :return:
        '''
        d = np.load(npz_file)
        self.train_x = d['train_x']
        self.train_y = d['train_y']
        self.valid_x = d['valid_x']
        self.valid_y = d['valid_y']
        self.test_x = d['test_x']
        self.test_y = d['test_y']

    def save_npz(self, npz_file='cinic10.npz'):
        '''
        save all data in npz to improve loading speed
        :param npz_file:
        :return:
        '''
        np.savez_compressed(npz_file, train_x=self.train_x, train_y=self.train_y, valid_x=self.valid_x,
                            valid_y=self.valid_y, test_x=self.test_x, test_y=self.test_y)

    def _readimages(self, img_paths):
        '''
        Multi-threaded loading to increase the loading speed from the dataset folder.
        :param img_paths:
        :return:
        '''
        def read(ls):
            for i in range(len(ls)):
                ls[i] = imageio.imread(ls[i])
                if ls[i].ndim == 2:
                    ls[i] = np.tile(ls[i][..., None], [1, 1, 3])

        group_ls = []
        ths = []
        batch_size = int(np.ceil(len(img_paths) / cpu_count()))
        for i in range(cpu_count()):
            ls = img_paths[i * batch_size: (i + 1) * batch_size]
            group_ls.append(ls)
            th = threading.Thread(target=read, args=(ls,))
            th.start()
            ths.append(th)

        for t in ths:
            t.join()

        new_ls = []
        for ls in group_ls:
            new_ls.extend(ls)

        return new_ls

    def shuffle(self, dataset_name='train'):
        '''
        shuffle all data
        :param dataset_name:
        :return:
        '''
        if dataset_name == 'train':
            ids = np.arange(len(self.train_x))
            np.random.shuffle(ids)
            self.train_x = np.array(self.train_x)[ids]
            self.train_y = np.array(self.train_y)[ids]

        elif dataset_name == 'valid':
            ids = np.arange(len(self.valid_x))
            np.random.shuffle(ids)
            self.valid_x = np.array(self.valid_x)[ids]
            self.valid_y = np.array(self.valid_y)[ids]

        elif dataset_name == 'test':
            ids = np.arange(len(self.test_x))
            np.random.shuffle(ids)
            self.test_x = np.array(self.test_x)[ids]
            self.test_y = np.array(self.test_y)[ids]

        else:
            raise RuntimeError('invalid dataset name ' + str(dataset_name))

    def get_train_batch_count(self, batch_size):
        '''
        Get the number of data sets in batches
        :param batch_size:
        :return:
        '''
        return int(np.ceil(len(self.train_x) / batch_size))

    def get_valid_batch_count(self, batch_size):
        '''
        Get the number of data sets in batches
        :param batch_size:
        :return:
        '''
        return int(np.ceil(len(self.valid_x) / batch_size))

    def get_test_batch_count(self, batch_size):
        '''
        Get the number of data sets in batches
        :param batch_size:
        :return:
        '''
        return int(np.ceil(len(self.test_x) / batch_size))

    def get_train_batch(self, batch_id, batch_size):
        '''
        Get a batch
        :param batch_id:
        :param batch_size:
        :return:
        '''
        return self.train_x[batch_id * batch_size: (batch_id + 1) * batch_size], \
               self.train_y[batch_id * batch_size: (batch_id + 1) * batch_size]

    def get_valid_batch(self, batch_id, batch_size):
        '''
        Get a batch
        :param batch_id:
        :param batch_size:
        :return:
        '''
        return self.valid_x[batch_id * batch_size: (batch_id + 1) * batch_size], \
               self.valid_y[batch_id * batch_size: (batch_id + 1) * batch_size]

    def get_test_batch(self, batch_id, batch_size):
        '''
        Get a batch
        :param batch_id:
        :param batch_size:
        :return:
        '''
        return self.test_x[batch_id * batch_size: (batch_id + 1) * batch_size], \
               self.test_y[batch_id * batch_size: (batch_id + 1) * batch_size]

    def data(self):
        '''
        Get all the data
        :return:
        '''
        return self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y


if __name__ == '__main__':
    ds = cinic10_dataset()

    if os.path.exists('datasets/cinic10.npz'):
        # load dataset from npz file
        ds.load_from_npz('datasets/cinic10.npz')
    else:
        # load from dataset dir
        ds.load('datasets/cinic10')
        # save to npz can improve next loading speed
        ds.save_npz('datasets/cinic10.npz')

    # get origin data
    train_x, train_y, valid_x, valid_y, test_x, test_y = ds.data()

    # or use some simple api
    print(ds.get_train_batch_count(100))
    print(ds.get_valid_batch_count(100))
    print(ds.get_test_batch_count(100))

    for b in range(ds.get_train_batch_count(100)):
        ds.get_train_batch(b, 100)

    for b in range(ds.get_valid_batch_count(100)):
        ds.get_valid_batch(b, 100)

    for b in range(ds.get_test_batch_count(100)):
        ds.get_test_batch(b, 100)
