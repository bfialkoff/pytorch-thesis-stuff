# this script deletes all the experiments that have less than a certain number of epochs
# just a pruning method
from pathlib import Path
from shutil import rmtree


if __name__ == '__main__':
    dl_dir = Path(__file__).joinpath('..', 'files', 'deep_learning')
    epoch_thresh = 25
    num_dirs_deleted = 0
    exp_dir = dl_dir.glob('*')
    for exp in exp_dir:
        num_epochs = len(list(exp.glob('weights/*')))
        if num_epochs < epoch_thresh:
            print('deleteing', exp)
            rmtree(exp)
            num_dirs_deleted +=1
    print('deleted', num_dirs_deleted, 'experiments')

