import torch.nn as nn
import torch.nn.functional as F

class downBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(downBlock, self).__init__()

        block = [   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class upBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upBlock, self).__init__()

        block = [   nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# The network with 9 blocks consists of:
# c7s1-32,d64,d128,R128,R128,R128,
# R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3

# c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
# dk denotes a 3×3 Convolution-InstanceNorm-ReLU layer with k filters, and stride 2.
# Rk denotes a residual block that contains two 3×3 convolutional layers with the same number of filters on both layer. 

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        # initial block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(3, 32, kernel_size=7, stride=1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True)]

        # down block
        model += [downBlock(32, 64)]
        model += [downBlock(64, 128)]

        # residual block
        num_of_resblock = 9
        for i in range(num_of_resblock):
            model += [ResidualBlock(128)]

        # up block
        model += [upBlock(128, 64)]
        model += [upBlock(64, 32)]

        # last block
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(32, 3, kernel_size=7),
                    nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# 70×70 PatchGAN
# Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, kernel_size=4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
        

