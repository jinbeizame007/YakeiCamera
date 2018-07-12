import argparse
from util import weights_init_normal
from util import ReplayBuffer
from models import Generator
from models import Discriminator
import torch
from torch import nn
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# initialization
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Optimizers & LR schedulers
optimizer_G = optim.Adam(itertools.chain(netG_X2Y.parameters(),netG_Y2X.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(netD_X.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(netD_Y.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# LR Scheduler
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, 0, opt.decay_epoch).step)

# Data
dataA = np.random.randn(shape=(3,256,256)).astype(np.float32)
dataB = np.random.randn(shape=(3,256,256)).astype(np.float32)
train = torch.utils.data.TensorDataset(torch.from_numpy(dataA), torch.from_numpy(dataB))
data_loader = torch.utils.data.DataLoader(train, batch_size=opt.batchSize, shuffle=True)
buffer_A = ReplayBuffer()
buffer_B = ReplayBuffer()

# targets
real_target = Tensor(opt.batchSize).fill_(1.0)
fake_target = Tensor(opt.batchSize).fill_(0.0)

# Parameters
lamnda_identity = 5.0
lambda_cycle = 10.0

for epoch in range(opt.n_epochs):
    for i, data in enumerate(data_loader):
        realA, realB = data

        ### Generator

        # Identity loss
        sameA = netG_B2A(realA)
        identity_loss_A = torch.nn.L1Loss(sameA, realA) * lamnda_identity
        sameB = netG_A2B(realB)
        identity_loss_B = torch.nn.L1Loss(sameB, realB) * lamnda_identity

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        GAN_loss_A2B = torch.nn.MSELoss(pred_fake, target_real)
        buffer_B.add(fake_B)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        GANloss_B2A = torch.nn.MSELoss(pred_fake, target_real)
        buffer_A.add(fake_A)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = torch.nn.L1Loss(recovered_A, real_A) * lambda_cycle

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = torch.nn.L1Loss(recovered_B, real_B) * lambda_cycle

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()


        ### Discriminator
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # real loss
        pred_real_A = netD_A(real_A)
        pred_real_B = netD_B(real_B)
        real_loss_A = torch.nn.MSELoss(pred_real_A, target_real)
        real_loss_B = torch.nn.MSELoss(pred_real_B, target_real)

        # fake loss
        fake_A = buffer_A.sample()
        fake_B = buffer_B.sample()
        pred_fake_A = netD_A(fake_A)
        pred_real_B = netD_B(fake_B)
        fake_loss_A = torch.nn.MSELoss(pred_fake_A, target_fake)
        fake_loss_B = torch.nn.MSELoss(pred_fake_B, target_fake)

        # total loss
        loss_D_A = real_loss_A + fake_loss_A
        loss_D_B = real_loss_B + fake_loss_B
        loss_D_A.backward()
        loss_D_B.backward()

        optimizer_D_A.step()
        optimizer_D_B.step()
    
    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'models/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'models/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'models/netD_A.pth')
    torch.save(netD_B.state_dict(), 'models/netD_B.pth')

