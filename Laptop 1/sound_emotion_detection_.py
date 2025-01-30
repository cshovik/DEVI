# Import necessary libraries

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

def detect():
    # Dummy detection logic
    # Replace this with the actual logic for sound and emotion detection
    results = {
        'emotion': 'neutral',  # Placeholder for actual emotion detection
        'sound': 'normal'      # Placeholder for actual sound detection
    }
    return results


# Disable warnings
warnings.filterwarnings('ignore')

# Function to download dataset using Kaggle API
def download_dataset():
    api = KaggleApi()
    api.authenticate()
    # Download and unzip the dataset
    api.dataset_download_files('ejlok1/toronto-emotional-speech-set-tess', path='./', unzip=True)

# Function to load dataset and create dataframe
def load_data():
    paths = []
    labels = []
    for dirname, _, filenames in os.walk('./tess toronto emotional speech set data'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        if len(paths) == 2800:
            break
    print('Dataset is Loaded')
    
    # Create dataframe
    df = pd.DataFrame({'speech': paths, 'label': labels})
    return df

# Function to visualize data
def visualize_data(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(df['label'])
    plt.title('Emotion Distribution in Dataset')
    plt.show()

# Function to display waveplot and spectrogram
def display_audio_features(df):
    for emotion in df['label'].unique():
        path = np.array(df['speech'][df['label'] == emotion])[0]
        data, sampling_rate = librosa.load(path)
        waveplot(data, sampling_rate, emotion)
        spectogram(data, sampling_rate, emotion)

# Helper function for waveplot
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(f"Waveplot for {emotion}", size=20)
    plt.plot(data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

# Helper function for spectrogram
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(f"Spectrogram for {emotion}", size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

# Function to extract MFCC features from audio
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Function to preprocess data and create model
def preprocess_and_train(df):
    # Extract MFCC features
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = np.array([x for x in X_mfcc])
    X = np.expand_dims(X, -1)
    
    # One-hot encode labels
    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']]).toarray()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=64)
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy * 100:.2f}%')
    
    # Plot training history
    plot_training_history(history)

# Function to plot training and validation loss and accuracy
def plot_training_history(history):
    epochs = range(len(history.history['accuracy']))
    plt.figure(figsize=(14, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main function
if __name__ == "__main__":
    # Step 1: Download dataset
    download_dataset()
    
    # Step 2: Load data
    df = load_data()
    
    # Step 3: Visualize data
    visualize_data(df)
    
    # Step 4: Display audio features
    display_audio_features(df)
    
    # Step 5: Preprocess data and train model
    preprocess_and_train(df)
