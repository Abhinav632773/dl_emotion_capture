# Speech Emotion Recognition using Python

A comprehensive deep learning project for analyzing and classifying emotions from speech audio files using LSTM neural networks(Long Short-Term Memory).
## what is LSTM 
It is a type of Recurrent Neural Network (RNN) architecture used in the field of deep learning, especially for processing and making predictions based on sequential data (time series of text, speech, etc.)

## 🎯 Project Overview

This project implements a Speech Emotion Recognition system that can classify audio files into different emotional categories. The model can detect and classify emotions in spoken words

## 📊 Dataset Information
- **Data set Link(kaggle) ->**https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- **Total Samples**: 800*7 audio files
- **Format**: WAV audio files
- **Target Words**: 200 different words spoken in the carrier phrase "Say the word _"
- **Audio Duration**: Capped at 3 seconds for consistent processing

### 🎭 Emotion Classes (7 categories)
- Anger
- Disgust
- Fear
- Happiness
- Surprise
- Sadness
- Neutral

Each emotion class contains 800 samples, ensuring balanced distribution across all categories.

## 🛠️ Requirements

### Libraries and Dependencies
```python
pandas, numpy, matplotlib, seaborn, librosa,
scikit-learn, tensorflow, keras, IPython
```

### Installation

pip install pandas numpy matplotlib seaborn librosa scikit-learn tensorflow keras ipython


## Data collection 
Here I used a famous data set socalled *"TESS Toronto emotional speech set data"*

## Feature Extraction
The project uses **MFCC (Mel-frequency cepstral coefficients)** for feature extraction

**LSTM-based Neural Network:**
```python
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
```


**Model Configuration:**
- **Input Shape**: (40, 1) - 40 MFCC features
- **LSTM Layer**: 256 units
- **Dense Layers**: 128 → 64 → 7 units
- **Dropout Rate**: 0.2 for regularization (to avoid overfitting)
- **Output**: 7 classes (emotions)
- **Activation**: Softmax for multi-class classification

### 4. Training Configuration
```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=50,
    batch_size=64
)
```

## 📈 Data Visualization

The project includes comprehensive visualization tools:

### Waveform Analysis
```python
def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()
```

### Spectrogram Analysis
```python
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
```

## 📊 Model Performance

- **Training Accuracy**: ~99.8%
- **Best Validation Accuracy**: 72.3%
- **Training Epochs**: 50
- **Batch Size**: 64

### Performance Insights
- **Lower pitched voices**: Appear as darker colors in spectrograms
- **Higher pitched voices**: Show brighter colors in frequency analysis
- **Overfitting observed**: High training accuracy vs. moderate validation accuracy

## 🙏 Acknowledgments

- **Dataset**: Toronto Emotional Speech Set (TESS)
- **Libraries**: librosa, TensorFlow, Keras, scikit-learn
- **Inspiration**: Advances in speech emotion recognition and affective computing

## 🔗 References

- [Librosa Documentation](https://librosa.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)

---

**Tags**: `#SpeechEmotionRecognition` `#Python` `#DeepLearning` `#LSTM` `#AudioProcessing` `#MachineLearning` `#Keras` `#TensorFlow`
