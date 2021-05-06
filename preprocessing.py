import musdb
import torch
import torchaudio
import random

#Creates a batch of audo samples randomly selected from Musdb
#This is currently slow, but memory efficient.
#I'll add a fast, but ram-heavy method in the future
def generateBatch(mus, sample_len=44100, size=100):
    X = torch.empty((size, sample_len))
    Y = torch.empty((size, sample_len))
    track = random.choice(mus.tracks)
    source = torch.from_numpy(track.audio.T[0])
    target = torch.from_numpy(track.targets['vocals'].audio.T[0])
    # source = torchaudio.load("/home/koko/code/audio-tools/Spirits.wav")[0][0]
    # target = torchaudio.load("/home/koko/code/audio-tools/SpiritsVocal.wav")[0][0]
    start = torch.randint(0, source.shape[0] - sample_len, (size,))
    for i in range(size):
        X[i] = source[start[i]:start[i]+sample_len]
        Y[i] = target[start[i]:start[i]+sample_len]
    return X, Y

#Generate Features
#Shape: [Sample, channel, frequency, time]
#channel0: Amplituide
#channel1: Phase
def generateFeatures(X, nfft=512):
    Xstft = torch.stft(X, nfft, onesided=True, return_complex=False)
    Xamp = torch.log(torch.sqrt(torch.pow(Xstft[:,:,:,0], 2) + torch.pow(Xstft[:,:,:,1], 2)) + 1)
    Xphase = torch.atan2(Xstft[:,:,:,1],Xstft[:,:,:,0])
    return torch.stack((Xamp,Xphase), 1)

#Dot product of input and output
def idealMask(X, Y):
    M = (X[:,0] * Y[:,0]) / (X[:,0] * X[:,0])
    M[torch.where(torch.isnan(M))] = 0
    M[torch.where(M > 1)] = 1
    return M

#Creates audio from Tensor 
def generateAudio(X):
    A = (torch.exp(X[:,0]) - 1) * torch.exp(1j * X[:,1])
    nfft = 2 * (A.shape[1]-1)
    A = torch.istft(A, nfft, onesided=True, return_complex=False)
    torchaudio.save("Test.wav", A[0].unsqueeze(0).float(), 44100)

def applyMask(X, mask):
    return torch.stack((X[:,0] * mask, X[:,1]), 1)