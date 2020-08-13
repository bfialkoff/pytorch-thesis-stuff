"""
This generator is for experiments in directly regressing force values from 1d samples, regresnet models
"""
from random import shuffle, sample
import threading

import pandas as pd
import numpy as np


class EmgSampleGenerator:
    def __init__(self, csv_features_path,  batch_size, feature_len=60, num_channels=8, is_train=True, is_debug=False):
        self.input_shape = (feature_len, num_channels)
        self.is_debug = is_debug
        if num_channels == 8:
            self.input_columns = [f'channel_{c}' for c in range(num_channels)]
        elif num_channels == 32:
            self.input_columns = [f'channel_{c}_imf_{imf}' for imf in range(4) for c in range(8)]

        self.output_column = 'mean_force'
        self.features = pd.read_csv(csv_features_path)

        self.features['sample_num'] = np.arange(len(self.features))
        self.sample_map = {v: i for i, v in enumerate(self.features['sample_num'].unique())}
        #self.sample_map = {v: i for i, v in enumerate(self.features['sample_num'].unique())}
        #self.features['sample_num'] = self.features['sample_num'].map(self.sample_map)

        self.labels = self.features[['sample_num', 'mean_force']]
        self.features = self.features.drop(columns=['subject', 'signal_num'], axis=1)
        self.is_train = is_train
        self.batch_size = batch_size
        self.val_index_list = self.get_index_list()
        self.index_list = self.get_index_list()
        self.num_samples = len(self.index_list)
        self.batch_size = self.num_samples
        self.steps = self.num_samples // self.batch_size
        self.features = self.features.set_index('sample_num')
        self.labels = self.labels.set_index('sample_num')
        self.gen = self.train_generator() if is_train else self.val_generator()
        self.lock = threading.Lock()

    def _get_index_list(self):
        index_list = list(self.sample_map.values())
        #index_list = self.features.index.values.tolist()
        remainder = len(index_list) % self.batch_size
        if remainder:
            missing_samples = self.batch_size - remainder
            samples_to_duplicate = self.safe_sample(index_list, missing_samples)
            index_list += samples_to_duplicate
        shuffle(index_list)
        return index_list


    def get_index_list(self):
        index_list = list(self.sample_map.values())
        if self.is_train:
            shuffle(index_list)
        return index_list

    def safe_sample(self, p, k):
        p_size = len(p)
        if k > p_size:
            s = self.safe_sample(p, p_size)
            s += self.safe_sample(p, k - p_size)
        else:
            s = sample(p, k)
        return s

    def get_input_outputs(self, batch_rows, debug_tag=''):
        inputs, outputs = batch_rows[self.input_columns].values, batch_rows[self.output_column].values
        outputs = outputs.reshape(-1)
        return inputs, outputs

    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                self.index_list = self.get_index_list()
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            batch_rows = self.features.loc[batch_indices]
            input_images, outputs = self.get_input_outputs(batch_rows)
            yield input_images, outputs

    def val_generator(self):
        # todo ensure that this yields one subject at a time, no shuffle, ensure all data is yielded
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True:
            counter += 1
            if counter == self.steps:
                counter = 0
                start_i = counter * self.batch_size
                end_i = start_i + remainder
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size
            batch_feature_rows = self.features.loc[sorted(self.val_index_list)[start_i:end_i]]
            batch_label_rows = self.labels.loc[sorted(self.val_index_list)[start_i:end_i]]
            input_images, outputs = self.get_input_outputs(batch_feature_rows, batch_label_rows)
            yield input_images, outputs


    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.gen)

if __name__ == '__main__':
    from pathlib import Path

    train_features_path = Path(__file__, '..', '..', '..', 'files', 'omp_generated_vanilla_annotations',
                               'val_features.csv').resolve()
    train_labels_path = Path(__file__, '..', '..', '..', 'files', 'omp_generated_vanilla_annotations', 'val_labels.csv').resolve()
    emg_gen = EmgSampleGenerator(train_features_path, 16)

    for i, d in emg_gen:
        print(d)
