from keras.callbacks import Callback
import h5py
import numpy as np
import keras
import  os
import os
import pickle
import json
import argparse
import keras
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model

matplotlib.use("Agg")

class MyCallBack(Callback):
    def __init__(self, state_dir):
        super(MyCallBack, self).__init__()

        self.state_dir = state_dir

        self.json_log_file = os.path.join(self.state_dir, 'logs.json')
        self.png_log_file = os.path.join(self.state_dir, 'logs.png')
        self.last_model_file = os.path.join(self.state_dir, 'last_model.h5')
        self.best_model_file = os.path.join(self.state_dir, 'best_model.h5')

        if os.path.exists(self.json_log_file):
            with open(self.json_log_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'epoch': -1,
                'best': 0
            }

    def get_initial_epoch(self):
        return self.history['epoch'] + 1

    def load_last_model(self):
        if os.path.exists(self.last_model_file):
            return load_model(self.last_model_file)
        else:
            return None

    def on_epoch_end(self, epoch, logs=None):
        self.history['epoch'] = epoch

        logs = logs or None
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.save_json_log()
        #self.save_png_log()
        self.save_last_model()
        self.save_best_model()

    def save_json_log(self):
        with open(self.json_log_file, 'w') as f:
            json.dump(self.history, f)

        print('save json log to {}'.format(self.json_log_file))

    def save_png_log(self):
        history = self.history
        size = history['epoch'] + 1

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, size), history["loss"], label="train_loss")
        plt.plot(np.arange(0, size), history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, size), history["acc"], label="train_acc")
        plt.plot(np.arange(0, size), history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.png_log_file)

        print('save png log to {}'.format(self.png_log_file))

    def save_last_model(self):
        self.model.save(self.last_model_file)
        print('save last model to {}'.format(self.last_model_file))

    def save_best_model(self):
        epoch = self.history['epoch']
        best = self.history['best']
        val_acc = self.history['val_acc']

        if val_acc[-1] > best:
            self.history['best'] = val_acc[-1]
            self.model.save(self.best_model_file)
            print('val_acc inc from {} to {}, save best model to {}'.format(best, val_acc[-1], self.best_model_file))
        else:
            print('no inc in val_acc ...')
