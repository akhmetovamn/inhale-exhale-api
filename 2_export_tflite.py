import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('inhale_exhale_classifier.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('inhale_exhale_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Exported inhale_exhale_classifier.tflite successfully!")

