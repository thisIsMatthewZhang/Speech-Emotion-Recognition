from data_augmentation import *
from feature_extraction import *
from model import create_model
import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library used to analyze and collect data from audio files.
import librosa

# seaborn is a data visualization library based on matplotlib.
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn is a machine learning library
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play audio files
import simpleaudio as sa
import sounddevice as sd

# keras is a deep learning API that makes it easier to implement neural networks
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Paths for data.
Ravdess = "audio_speech_actors_01-24"
Crema = "AudioWAV"
Tess = "TESS Toronto emotional speech set data"
Savee = "ALL"

# Ravdess
ravdess_directory_list = os.listdir(Ravdess)
# print(ravdess_directory_list)
file_emotion_r = []
file_path_r = []
for dir in ravdess_directory_list:
    # as there are 20 different actors in our previous directory we need to extract files for each actor.
    # print(dir)
    # stores the files of each actor directory
    if 'Actor' not in dir:
        continue
    actor = os.listdir(Ravdess + '/' + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion_r.append(int(part[2]))
        file_path_r.append(Ravdess + '/' + dir + '/' + file)

# dataframe of emotion of each file
emotion_df = pd.DataFrame(file_emotion_r, columns=['Emotions'])
# dataframe for path of files
path_df = pd.DataFrame(file_path_r, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions
Ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust',
                             8: 'surprise'}, inplace=True)

# Crema
crema_directory_list = os.listdir(Crema)
file_emotion__c = []
file_path_c = []

for file in crema_directory_list:
    file_path_c.append(Crema + '/' + file)
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion__c.append('sad')
    elif part[2] == 'ANG':
        file_emotion__c.append('angry')
    elif part[2] == 'DIS':
        file_emotion__c.append('disgust')
    elif part[2] == 'FEA':
        file_emotion__c.append('fear')
    elif part[2] == 'HAP':
        file_emotion__c.append('happy')
    elif part[2] == 'NEU':
        file_emotion__c.append('neutral')
    else:
        file_emotion__c.append('unknown')

# dataframe of emotion of each file
emotion_df_c = pd.DataFrame(file_emotion__c, columns=['Emotions'])
# dataframe for each file path
path_df_c = pd.DataFrame(file_path_c, columns=['Path'])
Crema_df = pd.concat([emotion_df_c, path_df_c], axis=1)

# TESS
tess_directory_list = os.listdir(Tess)
file_emotion_t = []
file_path_t = []
for dir in tess_directory_list:
    dir_files = os.listdir(Tess + '/' + dir)
    for file in dir_files:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part == 'ps':
            file_emotion_t.append('surprise')
        else:
            file_emotion_t.append(part)
        file_path_t.append(Tess + '/' + dir + '/' + file)
# emotion dataframe for TESS
emotion_df_t = pd.DataFrame(file_emotion_t, columns=['Emotions'])
# path dataframe for TESS
path_df_t = pd.DataFrame(file_path_t, columns=['Path'])
Tess_df = pd.concat([emotion_df_t, path_df_t], axis=1)

# Savee
savee_directory_list = os.listdir(Savee)
file_emotion_s = []
file_path_s = []
for file in savee_directory_list:
    file_path_s.append(Savee + '/' + file)
    part = file.split('.')[0]
    part = part.split('_')[1]
    if part[0] == 'a':
        file_emotion_s.append('angry')
    elif part[0] == 'd':
        file_emotion_s.append('disgust')
    elif part[0] == 'f':
        file_emotion_s.append('fear')
    elif part[0] == 'h':
        file_emotion_s.append('happy')
    elif part[0] == 'n':
        file_emotion_s.append('neutral')
    elif part[0] == 's' and part[1] == 'a':
        file_emotion_s.append('sad')
    elif part[0] == 's' and part[1] == 'u':
        file_emotion_s.append('surprise')
    else:
        file_emotion_s.append('unknown')
emotion_df_s = pd.DataFrame(file_emotion_s, columns=['Emotions'])
path_df_s = pd.DataFrame(file_path_s, columns=['Path'])
Savee_df = pd.concat([emotion_df_s, path_df_s], axis=1)

# create an aggregate dataframe using the four previously created dataframes
agg_df = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df])
agg_df.to_csv("agg_df.csv", index=False)

# plot the distribution of emotions
plt.title("Count of Emotions", size=16)
sns.countplot(x='Emotions', data=agg_df, palette=['#3D25BE', '#BE252D'])
plt.xlabel('Emotion', size=12)
plt.ylabel('Count', size=12)
sns.despine()
plt.show()

def create_waveplot(data, sr, emotion):
    """plotting the waveplots and spectrogram of the audio"""
    plt.figure(figsize=(10, 6))
    plt.title(f'Waveplot for audio with {emotion} emotion', size=16)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, emotion):
    """display spectrogram from data input"""
    # stft function converts the data into short-time fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 6))
    plt.title(f'Spectrogram for audio with {emotion} emotion', size=16)
    librosa.display.specshow(data=Xdb, sr=sr, x_axis='Time', y_axis='Hertz (hz)')
    plt.colorbar()
    plt.show()

def play_audio(data, sr):
    sd.play(data, sr)
    sd.wait()

def play_simple_audio(path):
    """ Generate audio using path to audio file"""
    wave_obj = sa.WaveObject.from_wave_file(path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def generate(emotion):
    """ Generate wave plot, spectrogram, and play corresponding audio """
    path = np.array(agg_df.Path[agg_df.Emotions == emotion])[1]  # same as 'path = np.array(agg_df.Path[agg_df['Emotions'] == emotion])[1]'
    data, sampling_rate = librosa.load(path)
    create_waveplot(data, sampling_rate, emotion)
    create_spectrogram(data, sampling_rate, emotion)
    play_simple_audio(path)

# taking any example and checking for techniques.
path = np.array(agg_df.Path)[5]
e = np.array(agg_df.Emotions)[5]
data, sample_rate = librosa.load(path)

# 1) Simple audio
create_waveplot(data=data, sr=sample_rate, emotion=e)
play_audio(data=data, sr=sample_rate)

# 2) Noise injection
noise_injected = noise(data)
create_waveplot(data=noise_injected, sr=sample_rate, emotion=e)
play_audio(data=noise_injected, sr=sample_rate)

# 3) Stretching
stretched = stretch(data)
create_waveplot(data=stretched, sr=sample_rate, emotion=e)
play_audio(data=stretched, sr=sample_rate)

# 4) Shifting
shifted = shift(data)
create_waveplot(data=shifted, sr=sample_rate, emotion=e)
play_audio(data=shifted, sr=sample_rate)

# 5) Pitch
pitched = pitch(data=data, sr=sample_rate)
create_waveplot(data=pitched, sr=sample_rate, emotion=e)
play_audio(data=pitched, sr=sample_rate)

f, emotions = [], []
for path, emotion in zip(agg_df.Path, agg_df.Emotions):
    features = get_features(path)
    for el in features:
        f.append(el)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        emotions.append(emotion)
# print(len(Y), len(Z), agg_df.Path.shape)

Features = pd.DataFrame(f)
Features['labels'] = emotions
Features.to_csv('features.csv', index=False)
print(Features.head())

# Data preparation - we now need to normalize and split our data for training and testing
data_retrieval = Features.iloc[:, :-1].values
label_vals = Features['labels'].values
# One-hot encoding: technique to represent categorical values as numerical values in our ML model
# As this is a multiclass classification problem we are one-hot encoding our label_vals
encoder = OneHotEncoder()
label_vals = encoder.fit_transform(np.array(label_vals).reshape(-1, 1)).toarray()
# splitting data
x_train, x_test, y_train, y_test = train_test_split(data_retrieval, label_vals, random_state=0, shuffle=True)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# making our data compatible to model
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = create_model(x_train)

rlrp = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

print(f'Accuracy of our model on test data: {model.evaluate(x_test, y_test)[1] * 100}%')

epochs = list(range(50))
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20, 6)
ax[0].plot(epochs, train_loss, label='Training Loss')
ax[0].plot(epochs, test_loss, label='Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs, train_acc, label='Training Accuracy')
ax[1].plot(epochs, test_acc, label='Testing Accuracy')
ax[1].set_title("Training & Testing Accuracy")
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

# predicting on test data
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()
df.head(10)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidths=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test, y_pred))

# the current model is most accurate at predicting angry and surprised emotions
# we can improve the accuracy with additional augmentation and feature extractions