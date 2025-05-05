import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from pydub import AudioSegment

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Paths
DATA_DIR = 'audio'
CLASSES = ['inhale', 'exhale']
FEATURES = []
LABELS = []

# Process each audio file
for idx, cls in enumerate(CLASSES):
    cls_folder = os.path.join(DATA_DIR, cls)
    for filename in os.listdir(cls_folder):
        if filename.lower().endswith('.wav'):
            filepath = os.path.join(cls_folder, filename)
            try:
                # Load and convert audio
                audio = AudioSegment.from_file(filepath)
                audio = audio.set_channels(1).set_frame_rate(16000)
                samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

                # Convert to tensor and run through YAMNet
                samples_tensor = tf.convert_to_tensor(samples)
                scores, embeddings, spectrogram = yamnet_model(samples_tensor)
                mean_embedding = tf.reduce_mean(embeddings, axis=0)

                FEATURES.append(mean_embedding.numpy())
                LABELS.append(idx)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Save extracted features
np.save('features.npy', np.array(FEATURES))
np.save('labels.npy', np.array(LABELS))

print("Preprocessing done! Saved features.npy and labels.npy.")
