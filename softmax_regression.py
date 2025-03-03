import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./data2/deeplearning.mplstyle')
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from data2.lab_utils_common import dlc
from data2.lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def softmax(z):
    ez = np.exp(z)
    return ez / np.sum(ez)

# # Visualize softmax function
# plt.close("all")
# plt_softmax(softmax)

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples = 2000, centers = centers, cluster_std = 1.0, random_state = 30)

# Convert data to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(centers), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")