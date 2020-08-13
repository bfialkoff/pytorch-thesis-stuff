"""
implement on plotting callback that inputs 2 generators,
performs inference on each signal in the val and train set
and plots the results and saves the image

"""
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from .plotter_base import PlotterBase

class PlotterCallback(PlotterBase):

    def __init__(self, train_generator, val_generator, summary_path, loss, model=None):
        PlotterBase.__init__(self,train_generator, val_generator, summary_path, loss, model=model)

    def sort_labels_predictions(self, label_force, predicted_force):
        label_force = label_force.reshape(-1)
        predicted_force = predicted_force.reshape(-1)

        label_inds = label_force.argsort()

        label_force = label_force[label_inds]
        predicted_force = predicted_force[label_inds]
        return label_force, predicted_force

    def get_force(self, features, coeffs, feature_len=30):
        features = np.split(features, len(features) // feature_len)
        predicted_force = np.array([np.dot(f, p[:-1]) + p[-1] for f, p in zip(features, coeffs)])
        return predicted_force

    def get_labels_predictions(self, train_labels, train_predictions, val_labels, val_predictions):
        train_features = self.train_generator.features.loc[self.train_generator.index_list]
        val_features = self.val_generator.features.loc[self.val_generator.index_list]

        train_labels = self.get_force(train_features, train_labels, feature_len=self.train_generator.feature_len)

        train_predictions = self.get_force(train_features, train_predictions, feature_len=self.train_generator.feature_len)

        val_labels = self.get_force(val_features, val_labels, feature_len=self.val_generator.feature_len)
        val_predictions = self.get_force(val_features, val_predictions, feature_len=self.val_generator.feature_len)


        train_labels, train_predictions = self.sort_labels_predictions(train_labels, train_predictions)
        val_labels, val_predictions = self.sort_labels_predictions(val_labels, val_predictions)

        return (train_labels, train_predictions), (val_labels, val_predictions)


    def run(self, gen_obj):
        y_true = np.zeros((gen_obj.steps * gen_obj.batch_size, gen_obj.num_params))
        y_pred = np.zeros((gen_obj.steps * gen_obj.batch_size, gen_obj.num_params))
        for i, (data, labels) in tqdm(enumerate(gen_obj.gen())):
            if i == gen_obj.steps:
                break
            start_index = i * gen_obj.batch_size
            end_index = start_index + gen_obj.batch_size
            pred = self.model.predict_on_batch(data)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred
        return y_true, y_pred

if __name__ == '__main__':
    from sys import exit

    loss_ax, train_ax, val_ax = PlotterCallback.get_gridspec()
    train_ax.set_title('train_ax')
    val_ax.set_title('val_ax')
    loss_ax.set_title('loss_ax')
    plt.show()
    exit()

    from image_builder import EmgImageGenerator
    from pathlib import Path
    from datetime import datetime
    train_path = Path(__file__, '..', 'files', 'train_annotations.csv')
    val_path = Path(__file__, '..', 'files', 'val_annotations.csv')
    date_id = datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__, '..', 'files', 'deep_learning', date_id).resolve()

    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    train_gen = EmgImageGenerator(train_path, 16)
    val_gen = EmgImageGenerator(val_path, 16)
    loss = lambda y, p: ((y - p) ** 2).mean()

    p = PlotterCallback(train_gen, val_gen, summary_path, loss)
    p.on_train_begin()
    p.on_epoch_end(1)
