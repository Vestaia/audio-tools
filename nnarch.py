import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time

#Convolutional neural net
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, 9, padding=4)
        self.conv2 = nn.Conv1d(64, 64, 9, padding=4)
        self.conv3 = nn.Conv1d(64, 128, 9, padding=4)
        self.conv4 = nn.Conv1d(128, 128, 9, padding=4)
        self.conv5 = nn.Conv1d(128, 256, 9, padding=4)
        self.conv6 = nn.Conv1d(256, 256, 9, padding=4)
        self.fc1 = nn.Linear(9472, 1024)
        #self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 1024)
        # self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 11)  
        
    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool1d(F.leaky_relu(self.conv2(x)), 3)
        x = F.leaky_relu(self.conv3(x))
        x = F.max_pool1d(F.leaky_relu(self.conv4(x)), 2)
        x = F.leaky_relu(self.conv5(x))
        x = F.avg_pool1d(F.leaky_relu(self.conv6(x)), 2)
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        # x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] 
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

BROADCAST_CHANNELS = 8
class Net2d(nn.Module):

    def __init__(self):
        super(Net2d, self).__init__()
        self.conv1 = nn.Conv2d(1, BROADCAST_CHANNELS, 1)
        self.conv2 = nn.Conv2d(BROADCAST_CHANNELS, BROADCAST_CHANNELS, 7, (1,3), (3,0))
        self.conv3 = nn.Conv2d(BROADCAST_CHANNELS, BROADCAST_CHANNELS, 7, (1,3), (3,0))
        self.conv4 = nn.Conv2d(BROADCAST_CHANNELS, BROADCAST_CHANNELS, 7, (1,3), (3,0))
        self.conv5 = nn.Conv2d(BROADCAST_CHANNELS, 1, 1)
        self.pool = nn.AvgPool2d((1,10),(1,10))
        self.fc = nn.Linear(513, 513)
        
        
    def forward(self, x):
        x = x[:,None]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#Loss function
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t)))


#High level API for NN
def learn(net, optimizer, X, y, batch_size = 50, device='cpu'):


    torch.backends.cudnn.fastest = True
    calc = LogCoshLoss()
    totalloss=0
    indicies = torch.randperm(len(X))
    for batch in range(np.int(np.floor(len(X)/batch_size))):
            optimizer.zero_grad()
            tx = X[indicies[batch*batch_size:(batch+1)*batch_size]].to(device)
            ty = y[indicies[batch*batch_size:(batch+1)*batch_size]].to(device)
            pred = net(tx)
            loss = calc(pred, ty)
            totalloss += loss.detach()
            loss.backward()
            optimizer.step()

    losses = totalloss/y.size(0) * batch_size * y.size(1)
    return losses.item()

def test(model, X, y, batch_size = 50, device='cpu'):

    calc = LogCoshLoss()
    model.eval()
    indicies = torch.randperm(len(X))
    pred = torch.zeros_like(y)

    for batch in range(np.int(np.floor(len(X)/batch_size))):
        tx = X[indicies[batch*batch_size:(batch+1)*batch_size]].to(device)
        pred[indicies[batch*batch_size:(batch+1)*batch_size]] = model(tx).detach().cpu()
    
    loss = calc(pred,y).detach().numpy() * y.size(1)
    return loss

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


#Trains model
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

    #Training
    print("Starting Training:")
    trainloss = []
    validation_loss = []
    plt.ion()
    start = time()
    
    epoch = 0
    # epoch = 
    for i in range(epochs):
        trainloss.append(learn(model, optimizer, trainX, trainy, batch_size=batch_size, device=device))
        elapsed = time() - start
        loss = test(model, validationX, validationy, batch_size=batch_size, device=device)
        validation_loss.append(loss)
        print("Epoch:", i+1)
        print("Train loss:", trainloss[i])
        print("validation loss:", validation_loss[i])
        print("Time:", elapsed)
        plot_train(trainloss,validation_loss)
        epoch += 1
