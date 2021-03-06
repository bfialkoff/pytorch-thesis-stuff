"""
This generator is for building channel_wise subtraction images using imfs
consequently the input shape yielded by this generator is
(batch_size, channel, channel, num_imfs)
"""
from random import shuffle, sample

import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from scipy.ndimage import convolve
import pandas as pd
from sklearn.utils import shuffle as sk_shuffle
from albumentations import (
    IAAPerspective, CLAHE, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, OneOf, Compose
)

from utils.permute_axis_subject import permute_axes_subtract

class EmgImageGenerator:
    def __init__(self, annotations, batch_size, num_channels=8, scaler=None, num_imfs=4, input_size=None, is_debug=False):
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
        self.input_size = input_size
        self.is_debug = is_debug
        self.imf_cols_dict = {f'imf_{imf}': [f'channel_{c}_imf_{imf}' for c in range(num_channels)] for
                              imf in range(num_imfs)}
        self.num_channels = num_channels
        self.num_imfs = num_imfs
        self.input_shape = (self.num_channels, self.num_channels, self.num_imfs)
        self.output_column = 'force'
        self.scaler = scaler
        self.input_images, self.outputs = self.process_annotations(annotations)
        self.batch_size = batch_size
        self.num_samples = len(self.outputs)
        self.steps = self.num_samples // self.batch_size
        self.index_list = list(range(self.num_samples))
        shuffle(self.index_list)

    # all this is unnecessary rms is a linear filter,
    # i could have left it as was, rms the df on __init__
    # and then permute_axes_subtract per batch
    def process_annotations(self, raw_annotations):
        kernel_length = 201
        rms_kernel_3d = np.ones(kernel_length) / kernel_length
        grouped_annotations = raw_annotations.groupby(['subject', 'signal_num'])
        # the proper length len(grouped_annotations)*(kernel_length - 1) + len(raw_annotations)
        after_rms_size = len(grouped_annotations)*(kernel_length - 1) + len(raw_annotations)
        batch_images, batch_outputs = np.zeros((after_rms_size, *self.input_shape)), np.zeros(after_rms_size)


        end_index = 0
        #total_len = 0
        for counter, (groupby_index, df) in enumerate(grouped_annotations):
            force_res = np.convolve(df[self.output_column], rms_kernel_3d)
            #total_len += len(force_res)
            start_index = end_index
            end_index = start_index + len(force_res)
            batch_outputs[start_index:end_index] = force_res
            for i, (imf, channel_cols) in enumerate(self.imf_cols_dict.items()):
                inputs = df[channel_cols].values
                # this row makes the voltage difference proportional to the channel (ai-aj) * ai
                # input_images = input_images * np.expand_dims(batch_rows[channel_cols], 2)
                input_images = permute_axes_subtract(inputs)
                for r in range(self.num_channels):
                    for c in range(self.num_channels):
                        if r != c:  # on the diagonal the entire value is zero, no need to waste computation
                            curr_signal = input_images[:, r, c]
                            curr_res = np.convolve(curr_signal, rms_kernel_3d)
                            batch_images[start_index:end_index, r, c, i] = curr_res
                        else:
                            batch_images[start_index:end_index, r, c, i] = 0

        if self.scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(batch_outputs.reshape(-1, 1))
        batch_outputs = self.scaler.transform(batch_outputs.reshape(-1, 1)).reshape(-1)

        return batch_images, batch_outputs

    def transform_func(self, batch_images):
        batch_maxes = batch_images.max(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
        normalized_batch = batch_images / batch_maxes
        return normalized_batch

    def resize_batch(self, batch_images):
        dst = np.array((len(batch_images), self.input_size, self.input_size, 3), dtype=batch_images.dtype)
        for i, img in enumerate(batch_images):
            resized = cv2.resize(img, (self.input_size, self.input_size), cv2.INTER_NEAREST)
            dst[i] = resized
        return dst

    def get_input_outputs(self, batch_rows, debug_tag=''):
        batch_images, batch_outputs = np.zeros((len(batch_rows), *self.input_shape)), batch_rows[
            self.output_column].values
        for i, (imf, channel_cols) in enumerate(self.imf_cols_dict.items()):
            inputs = batch_rows[channel_cols].values
            input_images = permute_axes_subtract(inputs)

            # this row makes the voltage difference proportional to the channel (ai-aj) * ai
            # input_images = input_images * np.expand_dims(batch_rows[channel_cols], 2)
            input_images = input_images.reshape(-1, self.num_channels, self.num_channels)
            batch_images[:, :, :, i] = input_images
        if self.input_size:
            self.resize_batch(batch_images)
        if self.is_debug:
            self.save_image(batch_images, debug_tag)
        return batch_images, batch_outputs,

    def save_image(self, images, debug_tag=''):
        images = images[:,:,:, 0:3]
        debug_dir = Path(__file__).joinpath('..', '..', '..', 'files', 'deep_learning', 'debug').resolve()
        if not debug_dir.exists():
            debug_dir.mkdir(parents=True)
        last_i = len(list(debug_dir.glob('*'))) + 1
        for i, img in enumerate(images):
            min_value = img.min()
            max_value = img.max() - min_value
            img = (img - min_value) / max_value
            img = (255 * img).astype(np.uint8)
            print(last_i + i, img.min(), img.max())
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            f_name = str(debug_dir.joinpath(f'{debug_tag}{last_i + i}.jpg').resolve())
            cv2.imwrite(f_name, img)

    def train_generator(self):
        counter = 0
        while True:
            if counter == self.steps:
                shuffle(self.index_list)
                counter = 0
            batch_indices = sample(self.index_list, self.batch_size)
            input_images, outputs = self.input_images[batch_indices], self.outputs[batch_indices]
            if self.is_debug:
                self.save_image(input_images, debug_tag='')
            yield input_images, outputs

    def val_generator(self):
        # todo ensure that this yields one subject at a time, no shuffle, ensure all data is yielded
        counter = 0
        remainder = self.num_samples % self.batch_size or self.batch_size
        while True:
            if counter == self.steps:
                start_i = counter * self.batch_size
                end_i = start_i + remainder
                counter = 0
            else:
                start_i = counter * self.batch_size
                end_i = start_i + self.batch_size
            counter += 1
            input_images, outputs = self.input_images[start_i:end_i], self.outputs[start_i:end_i]
            yield input_images, outputs


def show_2_images(img1, img2):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()


if __name__ == '__main__':
    from pathlib import Path

    train_path = Path(__file__, '..', '..', '..', 'files', 'emd_dl_train_annotations.csv').resolve()
    val_path = Path(__file__, '..', '..', '..', 'files', 'emd_dl_val_annotations.csv').resolve()
    train_annotations = pd.read_csv(train_path)
    print(train_annotations['subject'].unique())
    val_annotations = pd.read_csv(val_path)
    print(val_annotations['subject'].unique())
    train_emg_gen = EmgImageGenerator(train_annotations, 16, num_imfs=4, is_debug=False)
    val_emg_gen = EmgImageGenerator(val_annotations, 16, num_imfs=4, is_debug=False)
    for i, d in train_emg_gen.train_generator():
        print(i)

