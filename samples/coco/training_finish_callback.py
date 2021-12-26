# 保存训练点
from keras.callbacks import Callback


class TrainingResultCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(TrainingResultCheckpoint, self).__init__()

        self.filepath = filepath
        self.new_file_override = True

    def on_train_end(self, logs=None):
        self.model.save_weight(self.filepath, overwrite=True)

        pass
