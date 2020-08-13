from torch import nn

class _Dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(_Dense, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.bnorm = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.bnorm(x)
        x = nn.ReLU()(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, repetitions=2):
        super(DenseBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.repetitions = repetitions
        self.dense_blocks = nn.ModuleList([_Dense(self.input_size, self.output_size)]
                      + [_Dense(self.output_size, self.output_size)
                         for _ in range(self.repetitions - 1)])
        self.b_norm1 = nn.BatchNorm1d(self.output_size)
        self.b_norm2 = nn.BatchNorm1d(self.output_size)
        self.linear_1 = nn.Linear(self.output_size, self.output_size)
        self.linear_2 = nn.Linear(self.input_size, self.output_size)



    def forward(self, x):
        identity = x

        for l in self.dense_blocks:
            x = l(x)

        x = self.linear_1(x)
        x = self.b_norm1(x)

        shortcut = self.linear_2(identity)
        shortcut = self.b_norm2(shortcut)

        x = shortcut + x
        x = nn.ReLU()(x)
        return x

class IdentityBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(IdentityBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(self.input_size, self.output_size)
        self.linear_2 = nn.Linear(self.input_size, self.output_size)
        self.linear_3 = nn.Linear(self.input_size, self.output_size)
        self.bnorm_1 = nn.BatchNorm1d(self.input_size)
        self.bnorm_2 = nn.BatchNorm1d(self.output_size)
        self.bnorm_3 = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        input_tensor = x
        x = self.linear_1(x)
        x = self.bnorm_1(x)
        x = nn.ReLU()(x)

        x = self.linear_2(x)
        x = self.bnorm_2(x)
        x = nn.ReLU()(x)

        x = self.linear_3(x)
        x = self.bnorm_3(x)
        x = x + input_tensor
        x = nn.ReLU()(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResnetBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dense_block = DenseBlock(self.input_size, self.output_size)
        self.identity_block_1 = IdentityBlock(self.output_size, self.output_size)
        self.identity_block_2 = IdentityBlock(self.output_size, self.output_size)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.identity_block_1(x)
        x = self.identity_block_2(x)
        return x

class RegResNet(nn.Module):

    def __init__(self, input_size, block_width, repititions=3):
        super(RegResNet, self).__init__()
        self.input_size = input_size
        self.repititions = repititions
        self.block_width = block_width
        self.resnet_blocks = nn.ModuleList([ResnetBlock(self.input_size, self.block_width)]
                                           + [ResnetBlock(self.block_width, self.block_width)
                                              for _ in range(self.repititions - 1)])
        self.bnorm = nn.BatchNorm1d(self.block_width)
        self.out = nn.Linear(self.block_width, 1)

    def forward(self, x):
        for layer in self.resnet_blocks:
            x = layer(x)

        x = self.bnorm(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    import numpy as np
    from torch import from_numpy
    model = RegResNet(8, 16)

    #r_block = ResnetBlock(8, 16)
    #d_block = _Dense(8, 16)

    arr = np.array([8 * [1]])
    arr1 = np.array([8 * [1], 8 * [5]])

    t = from_numpy(arr).float()
    t1 = from_numpy(arr1).float()

    model.eval()
    from torchsummary import summary
    summary(model, (8,))
    print(model(t1))

