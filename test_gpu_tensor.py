import tensorflow as tf
import numpy as np
import time

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check if GPU is available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow is using GPU: {gpus[0].name}")
else:
    print("TensorFlow is not using GPU. Running on CPU.")

# Generate synthetic dataset
input_dim = 1000
output_dim = 10
num_samples = 100000

X = np.random.random((num_samples, input_dim)).astype(np.float32)
y = np.random.randint(0, output_dim, num_samples).astype(np.int32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and measure time
batch_size = 128
epochs = 5

print("Starting TensorFlow training...")
start_time = time.time()
with tf.device('/GPU:0' if gpus else '/CPU:0'):  # Ensure GPU is explicitly used if available
    model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)  # Suppressed logging
end_time = time.time()

print(f"TensorFlow training completed in {end_time - start_time:.2f} seconds")

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Final Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")


#### /usr/bin/python3 /Users/arikpeled/PycharmProjects/ArikFinalProject/test_gpu_pytorch.py
# PyTorch is using Metal Performance Shaders (MPS) for GPU acceleration.
# Starting PyTorch training...
# PyTorch training completed in 8.85 seconds
# Final Loss: 2.0823, Accuracy: 0.3770
############
# TensorFlow is using GPU: /physical_device:GPU:0
# Starting TensorFlow training...
# TensorFlow training completed in 19.50 seconds
# Final Loss: 2.3026, Accuracy: 0.1013
