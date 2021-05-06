import torch
import torch.optim
import torchaudio
import nnarch
import random
from torchsummary import summary

test = torch.ones((10, 513, 610))
net = nnarch.audioSep().cuda()
summary(net, ( 513, 610))
print(net(test.cuda()).shape)