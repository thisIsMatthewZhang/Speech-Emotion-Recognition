# Speech Emotion Recognition (SER)
This project implements a robust Speech Emotion Recognition system using Deep Learning. It aggregates multiple famous speech datasets, performs audio data augmentation, extracts relevant acoustic features, and trains a 1D Convolutional Neural Network (CNN) to classify human emotions from audio clips.

## Datasets Used
The model is trained on an aggregate of four major emotional speech datasets:

RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song.

CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.

TESS: Toronto Emotional Speech Set.

SAVEE: Surrey Audio-Visual Emotional Expressed.

## Tech Stack
Audio Analysis: librosa, simpleaudio, sounddevice

Data Manipulation: pandas, numpy

Visualization: seaborn, matplotlib

Deep Learning: TensorFlow/Keras

Machine Learning: scikit-learn

## Pipeline Workflow
1. Data Preprocessing & Augmentation
The script normalizes emotion labels across all four datasets (e.g., Happy, Sad, Angry, Fear, etc.). To increase model robustness, several Data Augmentation techniques are applied:

Noise Injection: Adding random white noise to the signal.

Stretching: Changing the speed of the audio without affecting the pitch.

Shifting: Moving the audio forward or backward in time.

Pitch Shifting: Altering the pitch of the audio.

2. Feature Extraction
The system extracts acoustic features (likely MFCCs, Chroma, or Mel-spectrograms via the imported get_features function) from both raw and augmented audio files to create a comprehensive feature matrix stored in features.csv.

3. Model Architecture
The project utilizes a 1D Convolutional Neural Network (CNN) built for sequence data:

Input Layer: Takes in scaled acoustic features.

Hidden Layers: Includes Conv1D, MaxPooling1D, BatchNormalization, and Dropout layers to prevent overfitting.

Output Layer: A Dense layer with Softmax activation for multi-class emotion classification.

4. Training & Optimization
Batch Size: 64

Epochs: 60

Callback: ReduceLROnPlateau to dynamically lower the learning rate when the loss plateaus.

## Evaluation
The script generates several visualizations to analyze performance:

Waveplots & Spectrograms: Visual representation of audio signals in the time and frequency domains.

Learning Curves: Plots for Training/Testing Accuracy and Loss over 60 epochs.

Confusion Matrix: A heatmap visualization comparing predicted vs. actual labels across all emotion categories.

Classification Report: Detailed metrics including Precision, Recall, and F1-Score.

Note: The current model shows peak accuracy in predicting Angry and Surprised emotions.
