"""
this callback
"""
import os
import json
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, matthews_corrcoef
import torch

class PlotterBase:

    def __init__(self, train_generator, val_generator, summary_path, loss, sample_size=1, model=None, mode='regression'):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.summary_path = summary_path
        self.loss = loss
        self.sample_size = sample_size
        self.on_epoch_end = self.regression_on_epoch_end if mode == 'regression' else self.classification_on_epoch_end

    def regression_get_metrics(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        abs_res = np.abs(y_true - y_pred)
        mean_error = abs_res.mean()
        max_error = abs_res.max()
        std_error = abs_res.std()
        return loss, mean_error, max_error, std_error

    def classification_get_metrics(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        matthews = matthews_corrcoef(y_true, y_pred)
        return loss, accuracy, recall, precision, matthews



    def regression_run(self, model, gen_obj):
        raise NotImplementedError

    def classification_run(self, model, gen_obj):
        y_true = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size)  # here
        y_pred = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size)  # here
        for i, (data, labels) in enumerate(gen_obj.val_generator()):
            if i == gen_obj.steps:
                break
            data = torch.from_numpy(data).float()
            start_index = i * gen_obj.batch_size
            end_index = start_index + len(data)
            pred = model(data)
            pred = torch.nn.Sigmoid()(pred)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred.detach().numpy().reshape(-1)
        return y_true, y_pred


    def regression_on_epoch_end(self, epoch, model, logs=None):
        train_labels, train_predictions = self.regression_run(model, self.train_generator)
        val_labels, val_predictions = self.regression_run(model, self.val_generator)
        train_loss, train_mean_error, train_max_error, train_std_error = self.regression_get_metrics(train_labels,
                                                                                          train_predictions)
        val_loss, val_mean_error, val_max_error, val_std_error = self.regression_get_metrics(val_labels, val_predictions)

        update = {
                  'train_loss': train_loss.astype(float),
                  'train_mean_error': train_mean_error.astype(float),
                  'train_max_error': train_max_error.astype(float),
                  'train_std_error': train_std_error.astype(float),
                  'val_loss': val_loss.astype(float),
                  'val_mean_error': val_mean_error.astype(float),
                  'val_max_error': val_max_error.astype(float),
                  'val_std_error': val_std_error.astype(float)
                  }
        print('train_loss', update['train_loss'], 'val_loss', update['val_loss'])
        self.write_summary(epoch, update)
        self.regression_write_graph(epoch, train_labels, train_predictions, val_labels, val_predictions)

    def threshold(self, arr, thresh):
        arr[arr > thresh] = 1
        arr[arr < thresh] = 0
        return arr


    def classification_on_epoch_end(self, epoch, model, logs=None):
        train_labels, train_predictions = self.classification_run(model, self.train_generator)
        val_labels, val_predictions = self.classification_run(model, self.val_generator)

        train_predictions = self.threshold(train_predictions, 0.5)
        val_predictions = self.threshold(val_predictions, 0.5)

        train_loss, train_accuracy, train_recall, train_precision, train_matthews = self.classification_get_metrics(train_labels,
                                                                                          train_predictions)
        val_loss, val_accuracy, val_recall, val_precision, val_matthews = self.classification_get_metrics(val_labels, val_predictions)

        update = {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_recall': train_recall,
            'train_precision': train_precision,
            'train_matthews': train_matthews,

            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_recall': val_recall,
            'val_precision': val_precision,
            'val_matthews': val_matthews,
                }
        print('train_loss', update['train_loss'], 'val_loss', update['val_loss'])
        summary = self.write_summary(epoch, update)
        self.classification_write_graph(summary)

    def on_train_begin(self, model, logs=None):
        """
        if summary_path_dir doesnt exist create dir call write_sumamry
        """
        self.graph_path = self.summary_path.parents[0].joinpath('graphs').resolve()
        self.model_summary = self.summary_path.parents[0].joinpath('model.txt').resolve()
        if not self.graph_path.exists():
            self.graph_path.mkdir(parents=True)
        if not self.summary_path.exists():
            self.save_summary({})
        if model is not None and False: # fixme this doesnt work for torch models
            with open(self.model_summary, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

    @classmethod
    def load_summary(cls, summary_path):
        with open(summary_path, 'r') as f:
            s = json.load(f)
        return s

    def save_summary(self, summary):
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4, sort_keys=True)

    def write_summary(self, key, update):
        summary = self.load_summary(self.summary_path)
        summary.update({f'{key:02d}': update})
        self.save_summary(summary)
        return summary

    def get_losses(self):
        summary = self.load_summary(self.summary_path)
        num_epochs = len(summary)
        epochs, train_loss, val_loss = np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)
        for i, (epoch, v) in enumerate(summary.items()):
            epochs[i] = int(epoch)
            train_loss[i] = v['train_loss']
            val_loss[i] = v['val_loss']
        sorted_inds = epochs.argsort()
        epochs = epochs[sorted_inds]
        train_loss = train_loss[sorted_inds]
        val_loss = val_loss[sorted_inds]
        if num_epochs > 10 and False:
            epochs = epochs[10:]
            train_loss = train_loss[10:]
            val_loss = val_loss[10:]

        return epochs, train_loss, val_loss

    def get_labels_predictions(self, train_labels, train_predictions, val_labels, val_predictions):
        raise NotImplementedError

    def regression_write_graph(self, epoch, train_labels, train_predictions, val_labels, val_predictions):
        epochs, train_loss, val_loss = self.get_losses()

        loss_ax, train_ax, val_ax = self.get_regression_gridspec()
        ax_lim = [-0.9 ,1.2]

        train_ax.set_ylim(ax_lim)
        val_ax.set_ylim(ax_lim)
        loss_ax.set_title('loss')
        train_ax.set_title('train')
        val_ax.set_title('val')

        (train_labels, train_predictions), (val_labels, val_predictions) = self.get_labels_predictions(train_labels, train_predictions, val_labels, val_predictions)

        loss_ax.plot(epochs, train_loss, 'g', label='train')
        loss_ax.plot(epochs, val_loss, 'r', label='val', alpha=0.3)

        train_ax.plot(train_labels, 'g', label='label')
        train_ax.plot(train_predictions, 'r', alpha=0.2, label='prediction')
        val_ax.plot(val_labels, 'g', label='label')
        val_ax.plot(val_predictions, 'r', alpha=0.2, label='prediction')

        train_ax.legend()
        val_ax.legend()
        loss_ax.legend()
        plt.savefig(self.graph_path.joinpath(f'epoch_{epoch}.png').resolve())
        plt.close()

    def classification_write_graph(self, summary):
        epochs, train_loss, val_loss, train_acc, train_recall, train_precision, train_matthews = [], [], [], [], [], [], []
        val_acc, val_recall, val_precision, val_matthews = [], [], [], []
        for epoch, values in summary.items():
            epochs.append(int(epoch))
            train_loss.append(values['train_loss'])
            val_loss.append(values['val_loss'])
            train_acc.append(values['train_accuracy'])
            train_recall.append(values['train_recall'])
            train_precision.append(values['train_precision'])
            train_matthews.append(values['train_matthews'])
            val_acc.append(values['val_accuracy'])
            val_recall.append(values['val_recall'])
            val_precision.append(values['val_precision'])
            val_matthews.append(values['val_matthews'])

        loss_ax, accuracy_ax, recall_ax, precision_ax, matthews_ax = self.get_classification_gridspec()
        loss_ax.set_title('loss')
        accuracy_ax.set_title('accuracy')
        recall_ax.set_title('recall')
        precision_ax.set_title('precision')
        matthews_ax.set_title('matthews')


        loss_ax.plot(epochs, train_loss, 'g', label='train')
        loss_ax.plot(epochs, val_loss, 'r', label='val', alpha=0.3)

        accuracy_ax.plot(epochs, train_acc, 'g', label='label')
        recall_ax.plot(epochs, train_recall, 'g', label='label')
        precision_ax.plot(epochs, train_precision, 'g', label='label')
        matthews_ax.plot(epochs, train_matthews, 'g', label='label')

        accuracy_ax.plot(epochs, val_acc, 'r', label='prediction')
        recall_ax.plot(epochs, val_recall, 'r', label='prediction')
        precision_ax.plot(epochs, val_precision, 'r', label='prediction')
        matthews_ax.plot(epochs, val_matthews, 'r', label='prediction')

        accuracy_ax.legend()
        recall_ax.legend()
        precision_ax.legend()
        matthews_ax.legend()
        loss_ax.legend()
        plt.savefig(self.graph_path.joinpath(f'epoch_{epochs[-1]}.png').resolve())
        plt.close()


    @classmethod
    def get_regression_gridspec(cls):
        fig10 = plt.figure(constrained_layout=True)
        gs0 = fig10.add_gridspec(1, 2)
        loss_ax = gs0[0].subgridspec(1, 1).subplots()
        train_ax, val_ax = gs0[1].subgridspec(2, 1).subplots()
        return loss_ax, train_ax, val_ax

    @classmethod
    def get_classification_gridspec(cls):
        fig10 = plt.figure(constrained_layout=True)
        gs0 = fig10.add_gridspec(2, 1)
        loss_ax = gs0[0].subgridspec(1, 1).subplots()
        (accuracy_ax, recall_ax), (precision_ax, matthews_ax) = gs0[1].subgridspec(2, 2).subplots()
        return loss_ax, accuracy_ax, recall_ax, precision_ax, matthews_ax


    def run(self, model, gen_obj):
        raise NotImplementedError

    # fixme this is a divergence point try to abstract this and rename it to run
    #  one solution might be to accept another dimension for y_true and y_pred which
    #  will be 9 for omp and 1 for the direct force approaches and drop the reshape
    #  which will need to be handled later in plotter_callback but is already handled in omp_plotter_callback
    def _run(self, gen_obj):
        y_true = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size) # here
        y_pred = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size) # here
        for i, (data, labels) in tqdm(enumerate(gen_obj.val_generator())):
            if i == gen_obj.steps:
                break
            start_index = i * gen_obj.batch_size
            end_index = start_index + len(data)
            pred = self.model.predict_on_batch(data)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred.reshape(-1) # here
        return y_true, y_pred

    @classmethod
    def plot_summary(cls, summary_path):
        graph_path = summary_path.parent[0].joinpath('graphs').resole()
        graph_path.makedir()
        summary = cls.load_summary(summary_path)
        train_labels = summary['train_labels']
        val_labels = summary['val_labels']
        train_inds = train_labels.argsort()
        val_inds = val_labels.argsort()
        train_labels = np.array(train_labels[train_inds])
        val_labels = np.array(val_labels[val_inds])
        del summary['train_labels'], summary['val_labels']
        f, (ax1, ax2) = plt.subplots(1, 2)
        for k, v in summary.items():
            train_pred = np.array(v['train_prediction'])[train_inds]
            val_pred = np.array(v['val_prediction'])[val_inds]
            ax1.plot(train_labels, 'g')
            ax1.plot(train_pred, 'r')
            ax2.plot(val_labels, 'g')
            ax2.plot(val_pred, 'r')
            plt.savefig(graph_path.joinpath(f'epoch_{k}.png'))
            ax1.cla()
            ax2.cla()


if __name__ == '__main__':
    PlotterBase.get_classification_gridspec()
    plt.show()