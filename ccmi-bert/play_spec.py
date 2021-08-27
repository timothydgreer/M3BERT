import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
#from playsound import playsound
import sounddevice as sd
import soundfile as sf
import librosa 
import scipy
import scipy.io.wavfile
import sys
import argparse

def to_decibles(signal):
	# Perform short time Fourier Transformation of signal and take absolute value of results
	stft = np.abs(librosa.stft(signal))
	# Convert to dB
	D = librosa.amplitude_to_db(stft, ref = np.max) # Set reference value to the maximum value of stft.
	return D # Return converted audio signal

def plot_spec(D, sr, instrument):
	fig, ax = plt.subplots(figsize = (30,10))
	spec = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', ax=ax)
	ax.set(title = 'Spectrogram of ' + instrument)
	fig.colorbar(spec)




'''
y, sr = sf.read('./test.wav',dtype='float32')
y = y[:,1]
S = np.abs(librosa.stft(y))
mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
print(np.shape(mel_spec))
print(type(mel_spec[0][0]))
S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)
print(np.shape(S_inv))
print(type(S_inv))
y = librosa.griffinlim(S_inv)

sd.play(y,sr)
status = sd.wait()
'''


x = np.load(str(sys.argv[1]))
print(np.shape(x))
x = np.transpose(x)
#x = x[52:180]
x = x[12:32]
print(np.shape(x))

#x = np.transpose(x)

sr = 44100
window = 'hamming'
win_length=2048
hop_length=1024

x = np.float32(x)

print(np.shape(x))
print(type(x[0][0]))

#Want to use mel spectrogram?
#y2 = librosa.feature.inverse.mel_to_audio(x, sr=sr, hop_length=hop_length, win_length=win_length, window=window)

#Want to use MFCCs? This is faster then
y2 = librosa.feature.inverse.mfcc_to_audio(x,hop_length=hop_length, win_length=win_length, window=window)

#This is the same as inverse mel to audio I think, but a little faster
#S = librosa.feature.inverse.mel_to_stft(M=x, sr=sr)
#y2 = librosa.griffinlim(S)
print("ALRIGHT")


#scipy.io.wavfile.write('mywav.wav',44100,y2)

print(len(y2))
sd.play(y2,44100)
status = sd.wait()

#plot_spec(to_decibles(y), sr, 'Song')

