# 保存训练点
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
import keras


class MetaCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaCheckpoint, self).__init__(filepath,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=save_best_only,
                                             save_weights_only=save_weights_only,
                                             mode=mode,
                                             period=period)

        self.filepath = filepath
        self.new_file_override = True
        self.meta = meta or {'epochs': [], self.monitor: []}

        if training_args:
            self.meta['training_args'] = training_args

    def on_epoch_end(self, epoch, logs={}):
        # 只有在‘只保存’最优版本且生成新的.h5文件的情况下
        # if self.save_best_only:
        #     current = logs.get(self.monitor)
        #     if self.monitor_op(current, self.best):
        #         self.new_file_override = True
        #     else:
        #         self.new_file_override = False

        self.new_file_override = True

        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.new_file_override and self.epochs_since_last_save == 0:
            # 只有在‘只保存’最优版本且生成新的.h5文件的情况下 才会继续添加meta
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))


import yaml
import os

check_file_path = './checkpoints/pointnet.h5'


def load_meta(model_fname):
    ''' Load meta configuration
    '''
    meta = {}

    with h5py.File(model_fname, 'r') as f:
        meta_group = f['meta']

        meta['training_args'] = yaml.load(
            meta_group.attrs['training_args'])
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])

    return meta


def get_last_status(model):
    last_epoch = -1
    last_meta = {}
    if os.path.exists(check_file_path):
        model.load_weights(check_file_path)
        last_meta = load_meta(check_file_path)
        last_epoch = last_meta.get('epochs')[-1]
    return last_epoch, last_meta


# if os.path.exists(check_file_path):
#     last_epoch, last_meta = get_last_status(model)
#     checkpoint = MetaCheckpoint(check_file_path, monitor='val_acc',
#                                 save_weights_only=True, save_best_only=True,
#                                 verbose=1, meta=last_meta)
# else:
#     checkpoint = MetaCheckpoint(check_file_path, monitor='val_acc',
#                                 save_weights_only=True, save_best_only=True,
#                                 verbose=1)
