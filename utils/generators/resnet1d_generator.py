"""
This generator is for experiments introducing a time dimension to the force regression
consequently the input shape yielded by this generator is
(batch_size, time_depth, channel, channel, num_imfs)
"""
from random import shuffle, sample

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import pandas as pd

def multi_rms(signals, window_size=400, axis=0):
    rms_signals = np.apply_along_axis(lambda m: window_rms(m, window_size=window_size), axis=axis, arr=signals)
    return rms_signals

def window_rms(signal, window_size=400):
    signal_squared = np.power(signal, 2)
    window = np.ones(window_size) / float(window_size)
    rms = np.sqrt(np.convolve(signal_squared, window, 'same'))
    return rms

class ResNet1dGenerator:
    def __init__(self, annotations, batch_size, num_channels=8, input_scaler=None, label_scaler=None, num_imfs=4, signal_window_size=512, is_debug=False, mode='regression'):
        """

        :param csv_path: path to the annotations files
        :param batch_size: number of data points to yield per itereation
        :param num_channels:
            the number of EMG channels that comprise a single sample. this is defined in the annotations files
            and shouldnt be adjusted
        :param num_imfs:
            the number of IMFs chosen during the EMD preprocessing stage this is defined in the annotations files
            and shouldnt be adjusted
        :param input_size:
            leave unspecified unless there is a reason to require the base 8x8xnum_imfs to be updated
        :param is_debug:
            will save images into a debug directory. Should be False during training!
        """

        self.is_debug = is_debug
        self.mode = mode
        self.id_cols = ['subject', 'signal_num']
        if num_channels * num_imfs == 8:
            self.input_columns = [f'channel_{c}' for c in range(num_channels)]
        elif num_channels * num_imfs == 32:
            self.input_columns = [f'channel_{c}_imf_{imf}' for imf in range(4) for c in range(8)]
        self.output_column = 'force'
        self.value_cols = self.input_columns + [self.output_column]
        self.num_channels = num_channels
        self.num_imfs = num_imfs
        self.signal_window_size = signal_window_size
        self.input_scaler = input_scaler
        self.label_scaler = label_scaler
        self.annotations = self.process_annotations(annotations)
        self.batch_size = batch_size
        self.num_samples = int(self.annotations.index.max() + 1)
        # fixme try and resolve a way to ensure that we cover all the data points
        self.steps = self.num_samples // self.batch_size
        self.index_list = list(range(self.num_samples))

        self.val_index_list = [(i * self.batch_size, (self.batch_size) * (i + 1)) for i in
                       range(self.steps)]
        self.val_index_list.append(
            (self.steps * self.batch_size, self.num_samples)
            )
        self.input_size = (self.batch_size, self.num_imfs * self.num_channels, self.signal_window_size)

    def process_annotations(self, raw_annotations):
        # fixme instead of resampling handle the residual window separately
        #  the solution for the final window will be to incorporate the missing portion from the preceding window
        annotations = raw_annotations.fillna(0)
        #annotations = annotations.drop(columns=['is_valid'], axis=1)
        annotations = annotations.set_index(self.id_cols)
        data_cols = self.id_cols + ['segment_num'] + self.value_cols
        results = []
        start_seg_id = 0
        for subject, signal in annotations.index.unique():
            rmsed = multi_rms(annotations.loc[(subject, signal), self.value_cols].values)
            # calculate remainder and dst size

            residual = len(rmsed) % self.signal_window_size
            num_windows = len(rmsed) // self.signal_window_size
            whole_part = rmsed[:num_windows * self.signal_window_size]

            if residual:
                num_windows += 1
                prepend = rmsed[-self.signal_window_size:]
                whole_part = np.concatenate((whole_part, prepend))
            res_len = self.signal_window_size * num_windows
            res = np.zeros((res_len, len(data_cols)))
            segments = start_seg_id + np.arange(0, len(res)) // self.signal_window_size
            res[:, 0:2] = subject, signal
            res[:, 2] = segments
            res[:, 3:] = whole_part
            results.append(res)
            start_seg_id = segments.max() + 1

        result_array = np.vstack(results)
        annotations = pd.DataFrame(data=result_array, columns=data_cols)
        annotations['segment_num'] = annotations['segment_num'].astype(int)

        if self.input_scaler is None:
            self.input_scaler = MinMaxScaler(feature_range=(0, 1))
            self.input_scaler.fit(annotations[self.input_columns])

        annotations = annotations.set_index('segment_num', drop=True)
        return annotations

    def get_input_outputs(self, batch_rows):
        batch_inputs = batch_rows[self.input_columns].values
        batch_outputs = batch_rows[self.output_column].reset_index().groupby('segment_num').mean().values.reshape(-1)

        if self.input_scaler:
            batch_inputs = self.input_scaler.transform(batch_inputs)
        if self.label_scaler:
            batch_outputs = self.label_scaler.transform(batch_outputs.reshape(-1, 1)).reshape(-1)
        if self.mode == 'classification':
            thresh = 0.3
            batch_outputs[batch_outputs > thresh] = 1
            batch_outputs[batch_outputs < thresh] = 0
        #batch_inputs = batch_inputs.reshape((-1, 1, self.num_imfs * self.num_channels, self.signal_window_size))
        batch_inputs = batch_inputs.reshape((-1, self.num_imfs * self.num_channels, self.signal_window_size))
        batch_inputs = np.repeat(batch_inputs[:, np.newaxis], 3, axis=1)
        batch_inputs = batch_inputs ** 0.3
        return batch_inputs, batch_outputs

    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                shuffle(self.index_list)
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            batch_rows = self.annotations.loc[batch_indices]
            input_images, outputs = self.get_input_outputs(batch_rows)
            if self.is_debug:
                self.save_image(input_images, debug_tag='')
            yield input_images, outputs

    def val_generator(self):
        for start_i, end_i in self.val_index_list:
            batch_rows = self.annotations.loc[start_i:(end_i - 1)]
            input_images, outputs = self.get_input_outputs(batch_rows)
            yield input_images, outputs


    def _val_generator(self):
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True: # fixme loop over val_index_list
            counter += 1
            if counter == self.steps:
                counter = 0
                start_i = counter * self.batch_size
                end_i = start_i + remainder
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size - 1
            batch_rows = self.annotations.loc[start_i:end_i]
            input_images, outputs = self.get_input_outputs(batch_rows)
            yield input_images, outputs


def show_2_images(img1, img2):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()


if __name__ == '__main__':
    from pathlib import Path

    annotations_dir = Path(__file__, '..', '..', '..', 'files', 'omp_generated_emd_annotations').resolve()
    label_scaler_path = annotations_dir.joinpath('force_scaler.pkl')
    train_features_path = annotations_dir.joinpath('train_annotations.csv')
    train_df = pd.read_csv(train_features_path)
    train_emg_gen = ResNet1dGenerator(train_df, batch_size=256)

    for i, d in enumerate(train_emg_gen.train_generator()):
        img = d[0][0]
        img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
        img = img * np.array([0.5989, 0.3, 0.140]).reshape((1, -1))
        show_2_images(img, img)
        print(i)

