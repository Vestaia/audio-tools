import torch
import torch.optim
import nnarch
from preprocessing import *
from torchsummary import summary


#Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading DB...")
mus = musdb.DB(root="/home/koko/code/Database/musdb18")


#Create features for training
print("Generating Features")
with torch.no_grad():
    X, Y = generateBatch(mus, sample_len=33075, size=300)
    X = generateFeatures(X)
    Y = generateFeatures(Y)
    M = idealMask(X, Y)
print("")
length = X.shape[3]

#Training net
model = nnarch.audioSep().to(device)
summary(model, (X.shape[2],X.shape[3]))
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
nnarch.train(model, optim, X[:,0], Y[:,0,:,int(length/2)], validation_size=100, epochs=10, batch_size=50)



#Testing
import matplotlib.pyplot as plt
X, Y = generateBatch(mus, sample_len=44100*15, size=1)
X = generateFeatures(X).to(device)
Y = generateFeatures(Y).to(device)
M = idealMask(X, Y)

output = torch.zeros_like(X[:,0])
model.eval()
with torch.no_grad():
    for i in range(3000):
        output[:,:,i] = model(X[:,0,:,i:i+length])
        if (i % 100 == 0):
            print(i)
#audio = applyMask(X[:,:,:,int(length/2):], output[:,:,:-int(length/2)]).to('cpu')
output = output.to('cpu')
M = M.to('cpu')
X = X.to('cpu')
Y = Y.to('cpu')
generateAudio(torch.stack((output,X[:,1]),1))

fig = plt.figure(figsize=(10,15))
vmax = torch.max(X[0,0])/2
#Target plot
target = fig.add_subplot(311)
target.imshow(Y[0,0,:,int(length/2):],vmin=0, vmax=vmax)
target.set_aspect(aspect=10)
#Prediction plot
pred = fig.add_subplot(312)
pred.imshow(output[0],vmin=0,vmax=vmax)
pred.set_aspect(aspect=10)
#Source plot
source = fig.add_subplot(313)
source.imshow(X[0,0],vmin=0,vmax=vmax)
source.set_aspect(aspect=10)
plt.show()