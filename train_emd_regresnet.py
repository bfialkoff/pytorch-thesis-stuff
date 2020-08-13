from pathlib import Path
from datetime import datetime

from progressbar import progressbar
import torch
import torch.optim as optim
from torch import nn
from torch import from_numpy

from utils.pytorch_regresnet import RegResNet
from utils.callbacks.plotter_callback import PlotterCallback
from utils.generators.sample_generator import EmgSampleGenerator
from utils.losses.numpy_losses import *

feature_len = 60
num_channels = 32



def make_weights_dir(experiment_dir):
    weights_path = experiment_dir.joinpath('weights')
    if not weights_path.exists():
        weights_path.mkdir(parents=True)
    return weights_path


if __name__ == '__main__':
    # todo ideal would be to define the train metric as tf metrics in order to avoid reiterating over the train data
    train_features_path = Path(__file__, '..', 'files', 'omp_generated_emd_annotations', 'train_features.csv').resolve()
    train_labels_path = Path(__file__, '..', 'files', 'omp_generated_emd_annotations', 'train_labels.csv').resolve()
    val_features_path = Path(__file__, '..', 'files', 'omp_generated_emd_annotations', 'val_features.csv').resolve()
    val_labels_path = Path(__file__, '..', 'files', 'omp_generated_emd_annotations', 'val_labels.csv').resolve()

    date_id = datetime.now().strftime('%Y%m%d%H%M')
    #date_id = '202008021807'
    experiment_dir = Path(__file__, '..').joinpath('files', 'deep_learning', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    initial_epoch = 0
    batch_size = 128
    #initial_weights = experiment_dir.joinpath('weights', f'{initial_epoch}.hdf5').resolve()
    #initial_weights = initial_weights if num_gpus else None
    initial_weights = None
    activation = None
    train_emg_gen = EmgSampleGenerator(train_features_path, batch_size=batch_size, num_channels=num_channels)
    callback_train_emg_gen = EmgSampleGenerator(train_features_path, batch_size=batch_size, is_train=False, num_channels=num_channels)
    callback_val_emg_gen = EmgSampleGenerator(val_features_path, batch_size=batch_size, is_train=False, num_channels=num_channels)

    train_steps = 1 * train_emg_gen.num_samples // train_emg_gen.batch_size
    num_epochs = 1000
    loss = numpy_mse
    weights_dir = make_weights_dir(experiment_dir)

    model = RegResNet(num_channels, block_width=32, repititions=1)
    p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    clr = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-6, max_lr=1e-3, step_size_up=3*train_steps)
    p.on_train_begin(model)
    for epoch in range(num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        print(f'Epoch {epoch} of {num_epochs}')
        for i, data in enumerate(train_emg_gen):
            if i == train_steps:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = from_numpy(inputs).float()
            labels = from_numpy(labels).float()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('loss: %.3f' %(loss.item()))
        weights_path = weights_dir.joinpath(f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), weights_path)
        p.on_epoch_end(epoch, model)
    print('Finished Training')

