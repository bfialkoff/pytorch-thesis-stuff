# In this script i need a generator that yield 512 chuncks of signal along with mean force over 512
#
from pathlib import Path
from datetime import datetime
import pickle

import torch
import torch.optim as optim
from torch import nn
from torch import from_numpy
from torchvision.models import resnet50
import pandas as pd

from models import MSResNet
from utils.callbacks.plotter_callback import PlotterCallback
from utils.generators.resnet1d_generator import ResNet1dGenerator
from utils.losses.numpy_losses import *

num_channels = 8


def init_weights(m):
    try:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    except:
        print(m)
def make_weights_dir(experiment_dir):
    weights_path = experiment_dir.joinpath('weights')
    if not weights_path.exists():
        weights_path.mkdir(parents=True)
    return weights_path

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

class MyRes50(nn.Module):
    def __init__(self):
        super(MyRes50, self).__init__()
        self.resnet_50 = resnet50(pretrained=True)
        self.dense_1 = nn.Linear(1000, 256)
        self.dense_2 = nn.Linear(256, 64)
        self.dense_3 = nn.Linear(64, 32)
        self.dense_4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.resnet_50(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x



if __name__ == '__main__':
    # todo ideal would be to define the train metric as tf metrics in order to avoid reiterating over the train data
    #train_features_path = Path(__file__, '..', 'files', 'emd_dl_train_annotations.csv').resolve()
    #val_features_path = Path(__file__, '..', 'files', 'emd_dl_val_annotations.csv').resolve()
    annotations_dir = Path(__file__, '..', 'files', 'omp_generated_emd_annotations').resolve()
    label_scaler_path = annotations_dir.joinpath('force_scaler.pkl')
    train_features_path = annotations_dir.joinpath('train_annotations.csv')
    val_features_path = annotations_dir.joinpath('val_annotations.csv')

    initial_date = ''
    initial_epoch = -1
    num_epochs = 10000

    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__, '..').joinpath('files', 'deep_learning', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    weights_dir = make_weights_dir(experiment_dir)
    initial_weights = weights_dir.joinpath(f'epoch_{initial_epoch}.pth') if (initial_epoch > 0) else None

    batch_size = 256
    label_scaler = load_scaler(label_scaler_path)
    train_df = pd.read_csv(train_features_path)
    val_df = pd.read_csv(val_features_path)

    # fixme if changing to vanilla annotations move to 8 channels 1 imf
    train_emg_gen = ResNet1dGenerator(train_df, batch_size=batch_size, label_scaler=label_scaler)
    callback_train_emg_gen = ResNet1dGenerator(train_df, batch_size=batch_size, label_scaler=label_scaler)
    callback_val_emg_gen = ResNet1dGenerator(val_df, batch_size=batch_size, label_scaler=label_scaler)


    train_steps = train_emg_gen.steps
    callback_frequency = 1
    loss = numpy_mse

    #model = RegResNet(num_channels, block_width=64, repititions=3)
    #model = MSResNet(num_channels, num_classes=1)
    model = MyRes50()


    #model.apply(init_weights) # maybe this, or maybe its the learning rate
    if initial_weights:
        print('loading', initial_weights)
        model.load_state_dict(torch.load(initial_weights))

    p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-4, max_lr=1e-2, step_size_up=2*train_steps)
    p.on_train_begin(model)
    for epoch in range(initial_epoch + 1, num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        print(f'Epoch {epoch} of {num_epochs}')
        for i, data in enumerate(train_emg_gen.train_generator()): #
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
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
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

