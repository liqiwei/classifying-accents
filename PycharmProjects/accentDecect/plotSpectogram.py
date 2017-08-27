#methods processing the data were taken from http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/ 
#with the matricies outputted, I'm able to put them in sklearn MLP for classification

import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import os
import sys
import glob
import librosa.display

#displays badly formatted spectograms

def load_files(file_paths):
    raw_sounds=[]
    for paths in file_paths:
        X,sr=librosa.load(paths)
        raw_sounds.append(X)
    return raw_sounds

def plot_specgram(accent_class, raw_sounds):
    i=1
    fig = plt.figure(figsize=(10, 20), dpi=200)
    for n, f in zip(accent_class, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.tight_layout()
    plt.show()

def plot_log_specgram(accent_class, raw_sounds):
    i=1
    fig=plt.figure(figsize=(25,60),dpi=500)
    for n,f in zip(accent_class,raw_sounds):
        plt.subplot(10,1,i)
        D=librosa.logamplitude(np.abs(librosa.stft(f))**2,ref_power=np.max)
        librosa.display.specshow(D,y_axis='log')
        plt.title(n.title())
        plt.tight_layout()
        i+=1
    plt.show()

def extract_feature(file_name):
    X, sample_rate=librosa.load(file_name)
    stft=np.abs(librosa.stft(X))
    mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sample_rate).T,axis=0)
    mel=np.mean(librosa.feature.melspectrogram(X,sr=sample_rate).T,axis=0)
    contrast=np.mean(librosa.feature.spectral_contrast(S=stft,sr=sample_rate).T,axis=0)
    tonnetz=np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                            sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(parent_dir, sub_dirs, file_ext="*wav"):
    #where does 193 come from...figure out later lol
    features, labels, =np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)): #create a file path
            try:
                mfccs, chroma, mel, contrast, tonnetz=extract_feature(fn)
            except Exception as e:
                print ("Error encountered while parsing file: "),fn
                continue
            ext_features=np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features=np.vstack([features,ext_features])
    return np.array(features),np.array(labels,dtype=np.int)

def one_hot_encode(labels):
    n_labels=len(labels)
    n_unique_labels=n_labels
    one_hot_encode=np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arrange(n_labels),labels]=1
    return one_hot_encode

parent_dir='audioTestFiles'
sub_dirs=['hindi','mandarin','russian']
print("i am here")
np.set_printoptions(threshold=np.inf)
features,labels=parse_audio_files(parent_dir,sub_dirs)
print("i am still here")
print(features)


# file_paths=["audioTestFiles/hindi/hindi1.wav","audioTestFiles/mandarin/mandarin1.wav","audioTestFiles/russian/russian1.wav"]
# accent_class=["hindi","mandarin","russian"]
#
# raw_sounds=load_files(file_paths)
# print("spectrograms for accents")
# plot_log_specgram(accent_class, raw_sounds)
