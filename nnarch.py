import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

#Convolution block wrapper, autopadding
class convblk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='none'):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride = in_channels, out_channels, kernel_size, stride
        self.convolute = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.kernel_size//2)
        self.activate = activation_func(activation)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        x = self.convolute(x)
        x = self.activate(x)
        x = self.batchnorm(x)
        return x

class deconvblk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='none'):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride = in_channels, out_channels, kernel_size, stride
        self.convolute = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.kernel_size//2)
        self.activate = activation_func(activation)
        self.batchnorm = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x):
        x = self.convolute(x)
        x = self.activate(x)
        x = self.batchnorm(x)
        return x

class resblk(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='none'):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size, self.stride = in_channels, out_channels, kernel_size, stride
        self.convolute1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.kernel_size//2)
        self.convolute2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, self.kernel_size//2)
        self.activate1 = activation_func(activation)
        self.activate2 = activation_func(activation)
        self.batchnorm1 = nn.BatchNorm2d(self.in_channels)
        self.batchnorm2 = nn.BatchNorm2d(self.out_channels)
        self.propagate = nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.in_channels)
    def forward(self, x):
        res = self.batchnorm3(x)
        res = self.propagate(res)
        x = self.batchnorm1(x)
        x = self.activate1(x)
        x = self.convolute1(x)
        x = self.batchnorm2(x)
        x = self.activate2(x)
        x = self.convolute2(x)
        x += res
        return x


class audioSepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, depth=1, block=resblk, activation='none'):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Sequential(block(in_channels, out_channels, kernel_size, stride, activation),\
            *[block(out_channels, out_channels, kernel_size, 1, activation) for _ in range(depth - 1)])
      
    def forward(self, x):
        x = self.blocks(x)
            
        return x
        
#Sensitive Paramaters
CHANNEL_SCALE = 64
ACTIVATION = 'relu'
#Audio Seperation Network
class audioSep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = audioSepLayer(1, 1, 1, 1, 1, activation=ACTIVATION)
        self.layer2 = audioSepLayer(1, CHANNEL_SCALE, 7, 4, 3, activation=ACTIVATION)
        self.layer3 = audioSepLayer(CHANNEL_SCALE, CHANNEL_SCALE*2, 5, 3, 3, activation=ACTIVATION)
        self.layer4 = audioSepLayer(CHANNEL_SCALE*2, CHANNEL_SCALE*4, 5, 3, 3, activation=ACTIVATION)
        self.pool = nn.AvgPool2d(4,4)
        self.fc1 = nn.Linear(1024, 257)

    def forward(self, x):
        x = x[:,None]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#Loss function
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = 10* (y_t - y_prime_t)
        return torch.mean(torch.log(torch.cosh(ey_t)))


def learn(net, optimizer, X, y, batch_size=50, device='cpu'):

    torch.backends.cudnn.fastest = True
    calc = LogCoshLoss()
    net.train()
    totalloss=0
    indicies = torch.randperm(len(X))
    batches = np.int(np.floor(len(X)/batch_size))
    for batch in range(batches):
            optimizer.zero_grad()
            tx = X[indicies[batch*batch_size:(batch+1)*batch_size]].to(device)
            ty = y[indicies[batch*batch_size:(batch+1)*batch_size]].to(device)
            pred = net(tx)
            loss = calc(pred, ty)
            totalloss += loss.detach()
            loss.backward()
            optimizer.step()
    loss = totalloss/batches
    return loss.item()

def test(net, X, y, batch_size=50, device='cpu'):

    calc = LogCoshLoss()
    net.eval()
    totalloss=0
    batches = np.int(np.floor(len(X)/batch_size))
    for batch in range(batches):
        tx = X[batch*batch_size:(batch+1)*batch_size].to(device)
        ty = y[batch*batch_size:(batch+1)*batch_size].to(device)
        pred = net(tx)
        totalloss += calc(pred, ty).detach()
    loss = totalloss/batches
    return loss.item()

from IPython import display
def plot_train(trainloss, testloss):
    plt.clf()
    plt.plot(np.linspace(2,len(trainloss),len(trainloss)-1), trainloss[1:], label="train")
    plt.plot(np.linspace(2,len(testloss),len(testloss)-1), testloss[1:], label="test")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss') 
    display.display(plt.gcf())
    #display.clear_output(wait=True)


def train(model, optimizer, X, y, validation_size, batch_size=100, epochs=1):

    #Use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Scrambling data for training
    ind = np.random.permutation(len(y))

    #Splitting data to train and validation
    validationX = X[ind[:validation_size]]
    validationy = y[ind[:validation_size]]
    trainX = X[ind[validation_size:]]
    trainy = y[ind[validation_size:]]


    print("Starting Training:")
    trainloss = []
    validation_loss = []
    plt.ion()
    start = time()
    epoch = 0
    for i in range(epochs):
        trainloss.append(learn(model, optimizer, trainX, trainy, batch_size=batch_size, device=device))
        validation_loss.append(test(model, validationX, validationy, batch_size=batch_size, device=device))
        elapsed = time() - start
        print("Epoch:", i+1)
        print("Train loss:", trainloss[i])
        print("validation loss:", validation_loss[i])
        print("Time:", elapsed)
        plot_train(trainloss,validation_loss)
        epoch += 1

