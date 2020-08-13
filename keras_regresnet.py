"""
ResNet model for regression of Keras.
Optimal model for the paper

"Chen, D.; Hu, F.; Nian, G.; Yang, T. Deep Residual Learning for Nonlinear Regression. Entropy 2020, 22, 193."

Depth:28
Width:16
"""
from keras import layers, models
import keras

layer_type = 'bnorm'
reg = None # keras.regularizers.l1(l=10)
if layer_type == 'bnorm':
    l = layers.BatchNormalization
else:
    l = layers.Dropout


def identity_block(input_tensor, units):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        units:output shape
    # Returns
        Output tensor for the block.
    """
    x = layers.Dense(units, kernel_regularizer=reg)(input_tensor)
    x = l()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units, kernel_regularizer=reg)(x)
    x = l()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(units, kernel_regularizer=reg)(x)
    x = l()(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x


def dens_block(input_tensor, units, reps=2):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        unit: output tensor shape
    # Returns
        Output tensor for the block.
    """
    x = input_tensor

    for _ in range(reps):
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = l()(x)
        x = layers.Activation('relu')(x)

    x = layers.Dense(units, kernel_regularizer=reg)(x)
    x = l()(x)
    shortcut = layers.Dense(units, kernel_regularizer=reg)(input_tensor)
    x = l()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet_block(input_tensor, width=16):
    x = dens_block(input_tensor, width)
    x = identity_block(x, width)
    x = identity_block(x, width)
    return x

def RegResNet(input_size=8, reps=3, initial_weights=None, width=16, num_gpus=0, lr=1e-4):
    """
    Instantiates the RegResNet50 architecture.
    :param input_size: length of a single input sample
    :param reps: number of residual blocks to repeat
    :param initial_weights: path to initial weights
    :param width: number of hidden units a the dense layers of the residual block
    :param num_gpus: number of available gpus for distributed learning
    :param lr: learning rate
    :return: a keras model
    """
    input_layer = layers.Input(shape=(input_size,))
    x = input_layer
    for _ in range(reps):
        x = resnet_block(x, width)

    x = l()(x)
    x = layers.Dense(1, activation=None)(x)

    model = models.Model(inputs=input_layer, outputs=x)
    loss = keras.losses.mean_squared_error
    if num_gpus > 1:
        model = keras.utils.multi_gpu_utils.multi_gpu_model(model, gpus=num_gpus)
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(lr)
                  )
    if initial_weights:
        model.load_weights(initial_weights)
    return model

if __name__ == '__main__':
    model = RegResNet(8, reps=3, width=8, lr=1e-3)
    print(model.summary())