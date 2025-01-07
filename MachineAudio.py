import os
import json
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



# Load JSON file
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

# Process audio files and transcripts
def preprocess_data(data):
    audio_features = []
    transcripts = []

    for entry in data:
        audio_path = entry['audio_file']
        transcript = entry['transcript']

        # Load and process audio
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)  # Take the mean of MFCC features

        audio_features.append(mfccs)
        transcripts.append(transcript)

    return np.array(audio_features), transcripts

# Encode text transcripts to numerical format
def encode_transcripts(transcripts, tokenizer):
    return tokenizer.texts_to_sequences(transcripts)

# Create tokenizer for text


def create_tokenizer(transcripts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(transcripts)
    return tokenizer

# Build a speech recognition model using TensorFlow
def build_model(input_dim, output_dim, max_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Plot accuracy and loss
def plot_metrics(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":

    # Path to your JSON file
    json_path = "combined_dataset.json"

    # Load and preprocess data
    data = load_data(json_path)
    audio_features, transcripts = preprocess_data(data)

    # Tokenize transcripts
    tokenizer = create_tokenizer(transcripts)
    encoded_transcripts = encode_transcripts(transcripts, tokenizer)

    # Pad sequences to a fixed length

    max_len = max(len(seq) for seq in encoded_transcripts)
    padded_transcripts = pad_sequences(encoded_transcripts, maxlen=max_len, padding='post')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(audio_features, padded_transcripts, test_size=0.2, random_state=42)

    # Build the model
    input_dim = X_train.shape[1]
    output_dim = len(tokenizer.word_index) + 1
    model = build_model(input_dim, output_dim, max_len)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    # Save the model
    model.save("speech_recognition_model.h5")

    # Plot metrics
    plot_metrics(history)

    print("Model training complete and saved!")