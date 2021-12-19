# 保存训练点
from keras.callbacks import ModelCheckpoint


class TrainingResultCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):
        super(TrainingResultCheckpoint, self).__init__(filepath,
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

    def on_train_end(self, logs=None):
        self.model.save(self.filepath, overwrite=True)

        pass
