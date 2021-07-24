import pandas as pd
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    result = np.array([])
    mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 40) .T,axis = 0)
    result = np.hstack((result, mfccs))
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate) .T,axis = 0)
    result = np.hstack((result, chroma))
    mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate) .T,axis = 0)
    result = np.hstack((result, mel))
    return result


emotions = {
    '01' : 'neutral',
    '02' : 'calm',
    '03' : 'happy',
    '04' : 'sad',
    '05' : 'angry',
    '06' : 'fearful',
    '07' : 'disgust',
    '08' : 'surprised'
}

def gender(g):
    if int(g[0:2]) % 2 == 0:
        return 'female'
    else:
        return 'male'


import tqdm
from tqdm import tqdm
import glob
def load_data(test_size = 0.2):
    x, y = [], []
    for file in tqdm(glob.glob("C:/Users/DELL/Desktop/Mini Project code/RAVDESS Dataset/Actor_*/*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]] + '_' + gender(file_name.split("-")[-1])
        feature = extract_feature(file)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size = test_size, random_state = 1)


file = "C:/Users/DELL/Desktop/Mini Project code/RAVDESS Dataset/Actor_08/03-01-05-02-02-02-08.wav"
feature = extract_feature(file)
X_train, X_val, y_train, y_val = load_data(test_size = 0.2)
print(X_train)
print(X_val)
print(y_train)
print(y_val)


from sklearn.neural_network import MLPClassifier
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = 'adaptive', max_iter = 500)

model.fit(X_train, y_train)

import time
y_pred = model.predict(X_val)
y_pre = model.predict([feature])
print(y_pre)
time.sleep(5)

import webbrowser
if y_pre[0] == "neutral_male" or y_pre[0] == "neutral_female":
    webbrowser.open('https://www.youtube.com/watch?v=OKKr_hzWKLI')
elif y_pre[0] == "calm_male" or y_pre[0] == "calm_female":
    webbrowser.open('https://www.youtube.com/watch?v=wruCWicGBA4')
elif y_pre[0] == "happy_male" or y_pre[0] == "happy_female":
    webbrowser.open('https://www.youtube.com/watch?v=7_epZBJqfks')
elif y_pre[0] == "sad_male" or y_pre[0] == "sad_female":
    webbrowser.open('https://www.youtube.com/watch?v=h-3nt92UFZo')
elif y_pre[0] == "angry_male" or y_pre[0] == "angry_female":
    webbrowser.open('https://www.youtube.com/watch?v=UbiWyPbAMIQ')
elif y_pre[0] == "fearful_male" or y_pre[0] == "fearful_female":
    webbrowser.open('https://www.forbes.com/sites/joshsteimle/2016/01/04/14-ways-to-conquer-fear/?sh=5ca5a3911c48')
elif y_pre[0] == "disgust_male" or y_pre[0] == "disgust_female":
    webbrowser.open('https://www.psychologytoday.com/us/blog/smell-life/201202/taking-control-disgust')
elif y_pre[0] == "surprised_male" or y_pre[0] == "surprised_female":
    webbrowser.open('https://www.paulekman.com/universal-emotions/what-is-surprise/')
else:
    webbrowser.open('https://www.scienceofpeople.com/emotions-list/')
