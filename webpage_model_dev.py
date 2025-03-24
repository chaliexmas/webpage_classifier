import numpy as np
import os
import cv2
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Load dataset
def load_images_from_folder(folder, target_size=(200, 200)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return np.array(images)

print("Loading dataset...")
web_images = load_images_from_folder('datasets/websites')
web_labels = np.ones(web_images.shape[0])  # All labeled as websites

# Normalize and reshape
X_data = web_images.astype('float32') / 255.0
X_flat = X_data.reshape(X_data.shape[0], -1)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_flat)
joblib.dump(pca, "Website_PCA_model.pkl")
print("PCA model saved.")

# Reshape for CNN
X_cnn = X_data.reshape(-1, 200, 200, 1)

# Define CNN model
def cnn_model(input_shape):
    model = Sequential([
        InputLayer(input_shape),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

print("Building and training CNN model...")
model = cnn_model(X_cnn.shape[1:])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_cnn, web_labels, epochs=20, batch_size=32, verbose=1)

# Save CNN model
model.save("Website_CNN_model.h5")
print("CNN model saved.")
