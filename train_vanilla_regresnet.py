#In PyTorch, images are represented as [batch_size, channels, height, width]

from pathlib import Path
from datetime import datetime

from progressbar import progressbar
import torch
import torch.optim as optim
from torch import nn
from torch import from_numpy

from utils.pytorch_regresnet import RegResNet
from models import DenseModel
#from my_regresnet import ResNet
from utils.callbacks.plotter_callback import PlotterCallback
from utils.generators.sample_generator import EmgSampleGenerator
from utils.losses.numpy_losses import *

feature_len = 60
num_channels = 8



def make_weights_dir(experiment_dir):
    weights_path = experiment_dir.joinpath('weights')
    if not weights_path.exists():
        weights_path.mkdir(parents=True)
    return weights_path


if __name__ == '__main__':
    # todo ideal would be to define the train metric as tf metrics in order to avoid reiterating over the train data
    train_features_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'train_features.csv').resolve()
    train_labels_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'train_labels.csv').resolve()
    val_features_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'val_features.csv').resolve()
    val_labels_path = Path(__file__, '..', 'files', 'omp_generated_vanilla_annotations', 'val_labels.csv').resolve()

    initial_date = ''
    initial_epoch = -1
    num_epochs = 10000

    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__, '..').joinpath('files', 'deep_learning', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    weights_dir = make_weights_dir(experiment_dir)
    initial_weights = weights_dir.joinpath(f'epoch_{initial_epoch}.pth') if (initial_epoch > 0) else None

    batch_size = 256
    activation = None
    train_emg_gen = EmgSampleGenerator(train_features_path, batch_size=batch_size)
    callback_train_emg_gen = EmgSampleGenerator(train_features_path, batch_size=batch_size, is_train=False)
    callback_val_emg_gen = EmgSampleGenerator(val_features_path, batch_size=batch_size, is_train=False)


    train_steps = train_emg_gen.steps
    callback_frequency = 10
    loss = numpy_mse

    #model = RegResNet(num_channels, block_width=64, repititions=3)
    model = DenseModel(num_channels, 1, [128, 64, 32, 16, 8])
    if initial_weights:
        print('loading', initial_weights)
        model.load_state_dict(torch.load(initial_weights))

    p = PlotterCallback(callback_train_emg_gen, callback_val_emg_gen, summary_path, loss)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = optim.lr_scheduler.CyclicLR(optimizer,base_lr=1e-6, max_lr=1e-3, step_size_up=3*train_steps)
    p.on_train_begin(model)
    for epoch in range(initial_epoch + 1, num_epochs): # loop over the dataset multiple times
        running_loss = 0.0
        print(f'Epoch {epoch} of {num_epochs}')
        for i, data in enumerate(train_emg_gen):
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

