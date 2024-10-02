from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models import *
from dataset import *

imageSize = 32

from dataset import *
from models import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=False, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    opt = parser.parse_args()
    print(opt)



    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    manualSeed = 121
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
        raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)
    
    print(opt.cuda)

    # set CUDA and GPU
    cuda = 1
    nc = 1
    ns11 = 451
    batch_size = 64
    device = torch.device("cuda:0" if cuda else "cpu")

    print('device: ', device)


    # load dataset

    dataset = AntennaDataset(annotations_file="/home/mingdian/antenna_gan/result/s11_dB_20lgabs_selected.csv",
                             img_dir='/home/mingdian/antenna_gan/result/dataset_BW_sorted', transform=transform)


    from torch.utils.data import DataLoader

    all_data = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if cuda else "cpu")
    cuda = 1
    nc = 1
    ndf = 64
    ns11 = 451
    ngpu = 2


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)


    # Generator Code
    netG = Generator(ngpu=ngpu).to(device)
    netG.apply(weights_init)
    # if there is a pretrained model for generator
    if opt.netG:
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    # batch_size, nz, 1, 1

    # Discriminator
    # if there is a pretrained model for discriminator
    netD = Critic(ngpu=ngpu).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    # summary(netD, (nc, 32, 32), batch_size=10)


    # Simulator
    netS = Simulator(ngpu=ngpu).to(device)
    netS_path = 'netS_simulator_mse_0.8679.pth'
    netS.load_state_dict(torch.load(netS_path))
    print(netS)

    for param in netS.parameters():
        param.requires_grad = False

    # lr = 0.01 # for SGD
    lr = 0.0001  # for Adam
    beta1 = 0.5

    # defining the number of epochs
    n_epochs = 200
    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []

    # custom loss function
    def my_loss(output, target):
        loss = torch.mean(abs(target) * (output - target) ** 2)
        return loss

    criterion = nn.MSELoss()
    netD_criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 0.9
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler

    # split full dataset into training dataset and test dataset
    test_split = 0.2
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print('dataset_size: ', dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print('val_indices: ', len(val_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              sampler=valid_sampler)

    train_losses = []
    test_losses = []

    # get some fixed training images
    real_image_fixed, s11_fixed, title = next(iter(train_loader))
    # real_image_fixed = next(iter(train_loader))
    real_image_fixed = real_image_fixed.to(device)
    s11_fixed = s11_fixed.to(device)

    # define fixed_noise for check the training of GAN model
    fixed_noise = torch.randn(batch_size, nz, device=device)
    print('fixed_noise: ', fixed_noise.shape)
    print('s11_fixed: ', s11_fixed.shape)
    s11_fixed_noise = torch.cat((s11_fixed, fixed_noise), 1)
    
    
    vutils.save_image(real_image_fixed.detach(),
                      'S11_gan_real_samples.png', normalize=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))



 

    """ Test with Gaussian Plus"""
    
    
    ### test with Gaussian plus
    from scipy import signal
    
    cf1 = 57.5
    cf2 = 60
    
    for i in range(1):
    
        t1 = np.linspace(-1, 1, 451) - (cf1-55)/10
        i, q, e1 = signal.gausspulse(t1, fc=3, bw=1, retquad=True, retenv=True)
        e1 = -e1*15
    
        t2 = np.linspace(-1, 1, 451) - (cf2 - 55) / 10
        i, q, e2 = signal.gausspulse(t2, fc=3, bw=0.5, retquad=True, retenv=True)
        e2 = -e2 * 15
    
        e = e1 - 10
        # e = e1 + e2 - 5
    
    
        plt.plot(e, color='sandybrown')
        plt.show()
    
        s11 = [e.tolist(), e.tolist(), e.tolist(), e.tolist()]
        s11 = torch.FloatTensor(s11)
    
        # s11 = torch.reshape(s11, (4, 451, 1, 1))
    
        ## test model performance
    
        netG_path = 'S11_gan_new_netG_epoch_191.pth'
        netD_path = 'S11_gan_new_netD_epoch_191.pth'
    
        netG.load_state_dict(torch.load(netG_path))
        netD.load_state_dict(torch.load(netD_path))
    
    
        # mean_val = -6.6106
        # max_val = -2.3970065116882324
        # min_val = -67.73655700683594
        #
        #
        # s11 = (s11 - min_val) / (max_val - min_val) - (mean_val - min_val) / (max_val - min_val)
        # s11 = s11 / 10
    
        s11 = s11.to(device)
    
        noise = torch.randn(4, nz, device=device)
    
        s11_noise = torch.cat((s11, noise), 1)
    
        fake = netG(s11_noise)
        netS_output = netS(fake)
    
        s11 = s11.squeeze()
        # s11 = s11 * 10
        # s11 = (s11 + (mean_val - min_val) / (max_val - min_val)) * (max_val - min_val) + min_val
    
    
        fig, ((ax_0, ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6, ax_7)) = plt.subplots(2, 4)
    
        fake = fake.detach().to('cpu').numpy()
        netS_output = netS_output.detach().to('cpu').numpy()
        s11 = s11.detach().to('cpu').numpy()
    
        ax_0.imshow(fake[0].squeeze(), cmap='gray')
        ax_1.imshow(fake[1].squeeze(), cmap='gray')
        ax_2.imshow(fake[2].squeeze(), cmap='gray')
        ax_3.imshow(fake[3].squeeze(), cmap='gray')
    
        f = np.linspace(45, 65, 451, endpoint=True)
    
        ax_4.plot(f, s11[0])
        ax_4.plot(f, netS_output[0])
        ax_5.plot(f, s11[1])
        ax_5.plot(f, netS_output[1])
        ax_6.plot(f, s11[2])
        ax_6.plot(f, netS_output[2])
        ax_7.plot(f, s11[3])
        ax_7.plot(f, netS_output[3])
    
        ax_4.set_ylim([-50, 0])
        ax_5.set_ylim([-50, 0])
        ax_6.set_ylim([-50, 0])
        ax_7.set_ylim([-50, 0])
    
        plt.savefig('./data_gen/gaussian_plus_result.png')
    
        plt.show()
        plt.close()