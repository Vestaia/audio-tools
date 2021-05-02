import torch
import torch.optim
import torchaudio
import musdb
import nnarch
import random

def getSample(mus, duration=5.0):
    while True:
        track = mus.tracks[10]#random.choice(mus.tracks)
        track.chunk_duration = duration
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x = track.audio.T
        y = track.targets['vocals'].audio.T
        yield x, y

def generateBatch(mus, sample_len=5.0, batch_size=100):
    samples = getSample(mus, sample_len)
    it = 0
    X = torch.empty((batch_size, int(sample_len * 44100)))
    Y = torch.empty((batch_size, int(sample_len * 44100)))
    for x, y in samples:
        X[it] = torch.from_numpy(x[0]).to(X)
        Y[it] = torch.from_numpy(y[0]).to(Y)
        it += 1
        if it == batch_size:
            return X, Y

def generateFeatures(X, nfft=1024):
    Xstft = torch.stft(X, nfft, onesided=True, return_complex=False)
    Xamp = torch.log(torch.sqrt(torch.pow(Xstft[:,:,:,0], 2) + torch.pow(Xstft[:,:,:,1], 2)) + 1)
    Xphase = torch.atan2(Xstft[:,:,:,1],Xstft[:,:,:,0])
    return torch.stack((Xamp,Xphase), 1)

def idealMask(X, Y):
    mag = torch.linspace(0,1,11)
    masks = X.expand(11, X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    masks = torch.swapdims(masks, 0, 4)[:,:,0]
    masks = mag * masks
    masks = torch.swapdims(masks, 0, 3)
    idealMask = torch.min(torch.abs(Y[:,0] - masks), dim=0)[1] * 0.1
    return idealMask

# def idealBinaryMask(X, Y):
#     mask = torch.zeros_like(X)
#     mask[torch.abs(Y - X) < torch.abs(Y)] = 1
#     return mask

def generateAudio(X):
    A = (torch.exp(X[:,0]) - 1) * torch.exp(1j * X[:,1])
    nfft = 2 * (A.shape[1]-1)
    A = torch.istft(A, nfft, onesided=True, return_complex=False)
    torchaudio.save("Test.wav", A[0].unsqueeze(0).float(), 44100)

def applyMask(X, mask):
    return torch.stack((X[:,0] * mask, X[:,1]), 1)


print("Loading DB...")
mus = musdb.DB(root="/Users/koko/Documents/Development/Database/musdb18")

print("Generating Features")
with torch.no_grad():
    X, Y = generateBatch(mus, sample_len=2, batch_size=40)
    X = generateFeatures(X)
    Y = generateFeatures(Y)
    M = idealMask(X, Y)
print("")

length = X.shape[3]
model = nnarch.Net2d()
optim = torch.optim.Adam(model.parameters(), lr=3e-3)
nnarch.train(model, optim, X[:,0], M[:,:,int(length/2)], validation_size=0, epochs=40, batch_size=40)

#Testing
import matplotlib.pyplot as plt
X, Y = generateBatch(mus, sample_len=20, batch_size=1)
X = generateFeatures(X)
Y = generateFeatures(Y)
M = idealMask(X, Y)
mask = torch.zeros_like(X[:,0])
with torch.no_grad():
    for i in range(2000):
        mask[:,:,i] = model(X[:,0,:,i:i+length])
        if (i % 10 == 0):
            print(i)
output = applyMask(X[:,:,:,int(length/2):], mask[:,:,:-int(length/2)])
generateAudio(output)

plt.rcParams["figure.figsize"] = (8,8)
plt.imshow(M[0])
plt.imshow(mask[0])