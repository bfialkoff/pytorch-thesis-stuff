import keras
import numpy as np
import torch

from utils.regresnet import RegResNet

"""
RegResNet(
  (resnet_blocks): ModuleList(
    (0): ResnetBlock(
      (dense_block): DenseBlock(
        (dense_blocks): ModuleList(
          (0): _Dense(
            (linear): Linear(in_features=7, out_features=16, bias=True)
            (bnorm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): _Dense(
            (linear): Linear(in_features=16, out_features=16, bias=True)
            (bnorm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (b_norm1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (b_norm2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (linear_1): Linear(in_features=16, out_features=16, bias=True)
        (linear_2): Linear(in_features=16, out_features=16, bias=True)
      )
      (identity_block_1): IdentityBlock(
        (linear_1): Linear(in_features=16, out_features=16, bias=True)
        (linear_2): Linear(in_features=16, out_features=16, bias=True)
        (linear_3): Linear(in_features=16, out_features=16, bias=True)
        (bnorm_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bnorm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bnorm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (identity_block_2): IdentityBlock(
        (linear_1): Linear(in_features=16, out_features=16, bias=True)
        (linear_2): Linear(in_features=16, out_features=16, bias=True)
        (linear_3): Linear(in_features=16, out_features=16, bias=True)
        (bnorm_1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bnorm_2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (bnorm_3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (bnorm): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (out): Linear(in_features=16, out_features=1, bias=True)
)
"""

if __name__ == '__main__':
    model = RegResNet(7, 16, repititions=1)
    print(model)
    keras_model = keras.models.load_model('keras_regresnet_7_1_16.h5')
    keras_weights = keras_model.get_weights()
    model.resnet_blocks[0].identity_block_1.linear_1
    model.resnet_blocks[0].identity_block_1.bnorm_1
    model.resnet_blocks[0].identity_block_1.linear_1
    model.resnet_blocks[0].identity_block_1.bnorm_1
    model.resnet_blocks[0].identity_block_1.linear_1
    model.resnet_blocks[0].identity_block_1.bnorm_1


    model.resnet_blocks[0].identity_block_2.linear_1
    model.resnet_blocks[0].identity_block_2.bnorm_1
    model.resnet_blocks[0].identity_block_2.linear_1
    model.resnet_blocks[0].identity_block_2.bnorm_1
    model.resnet_blocks[0].identity_block_2.linear_1
    model.resnet_blocks[0].identity_block_2.bnorm_1

    model.resnet_blocks[0].dense_block.linear_1
    model.resnet_blocks[0].dense_block.bnorm_1
    model.resnet_blocks[0].dense_block.linear_2
    model.resnet_blocks[0].dense_block.bnorm_2

    for i, res_bock in enumerate(model.resnet_blocks[0]):
        layer_weights = keras_weights[i]
