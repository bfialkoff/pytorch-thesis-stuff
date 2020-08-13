
from matplotlib import pyplot as plt
import numpy as np
from torch import from_numpy

from .plotter_base import PlotterBase
class PlotterCallback(PlotterBase):

    def __init__(self, train_generator, val_generator, summary_path, loss, sample_size=1, model=None):
        PlotterBase.__init__(self, train_generator, val_generator, summary_path, loss, sample_size, model)

    def get_labels_predictions(self, train_labels, train_predictions, val_labels, val_predictions):
        train_inds = train_labels.argsort()
        val_inds = val_labels.argsort()
        train_labels = train_labels[train_inds]
        train_predictions = train_predictions[train_inds]

        val_labels = val_labels[val_inds]
        val_predictions = val_predictions[val_inds]

        return (train_labels, train_predictions), (val_labels, val_predictions)


    def run(self, model, gen_obj):
        y_true = np.zeros(gen_obj.num_samples)
        y_pred = np.zeros(gen_obj.num_samples)
        for i, (data, labels) in enumerate(gen_obj.val_generator()):
            data = from_numpy(data).float()

            start_index = i * gen_obj.batch_size
            end_index = start_index + len(data)
            pred = model(data)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred.detach().numpy().reshape(-1)
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
