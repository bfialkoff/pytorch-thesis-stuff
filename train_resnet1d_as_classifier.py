# In this script i need a generator that yield 512 chuncks of signal along with mean force over 512
#
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch import from_numpy
import pandas as pd

from models import MSResNet
from utils.callbacks.plotter_callback import PlotterCallback
from utils.callbacks.mixed_mode_plotter_base import PlotterBase
from utils.generators.resnet1d_generator import ResNet1dGenerator
from utils.losses.numpy_losses import *

num_channels = 32



def make_weights_dir(experiment_dir):
    weights_path = experiment_dir.joinpath('weights')
    if not weights_path.exists():
        weights_path.mkdir(parents=True)
    return weights_path


if __name__ == '__main__':
    # todo ideal would be to define the train metric as tf metrics in order to avoid reiterating over the train data
    train_features_path = Path(__file__, '..', 'files', 'emd_dl_train_annotations.csv').resolve()
    val_features_path = Path(__file__, '..', 'files', 'emd_dl_val_annotations.csv').resolve()

    initial_date = ''
    initial_epoch = -1
    num_epochs = 10000

    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__, '..').joinpath('files', 'deep_learning', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    weights_dir = make_weights_dir(experiment_dir)
    initial_weights = weights_dir.joinpath(f'epoch_{initial_epoch}.pth') if (initial_epoch > 0) else None

    batch_size = 64
    activation = None
    train_df = pd.read_csv(train_features_path)
    val_df = pd.read_csv(val_features_path)
    train_emg_gen = ResNet1dGenerator(train_df, batch_size=batch_size, mode='classification')
    label_scaler = train_emg_gen.label_scaler
    callback_train_emg_gen = ResNet1dGenerator(train_df, batch_size=batch_size, label_scaler=label_scaler, mode='classification')
    callback_val_emg_gen = ResNet1dGenerator(val_df, batch_size=batch_size, label_scaler=label_scaler, mode='classification')


    train_steps = train_emg_gen.steps
    callback_frequency = 1
    loss = numpy_mse

    #model = RegResNet(num_channels, block_width=64, repititions=3)
    model = MSResNet(num_channels, num_classes=1)
    if initial_weights:
        print('loading', initial_weights)
        model.load_state_dict(torch.load(initial_weights))

    p = PlotterBase(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss, mode='classification')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-6, max_lr=1e-3, step_size_up=3*train_steps)
    p.on_train_begin(model)
    for epoch in range(initial_epoch + 1, num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        print(f'Epoch {epoch} of {num_epochs}')
        for i, data in enumerate(train_emg_gen.train_generator()):
            if i == train_steps:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = from_numpy(inputs).float()
            labels = from_numpy(labels).float().unsqueeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = nn.Sigmoid()(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('loss: %.7f' %(loss.item()))

        if not ((epoch + 1) % callback_frequency):
            weights_path = weights_dir.joinpath(f'epoch_{epoch}.pth')
            torch.save(model.state_dict(), weights_path)
            model.eval()
            p.on_epoch_end(epoch, model)
            model.train()

    print('Finished Training')

