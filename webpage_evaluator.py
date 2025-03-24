from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import pandas as pd
import joblib

# Load the saved CNN model
cnn_model = load_model("Website_CNN_model.h5")
print("CNN model loaded successfully.")

# Load PCA transformer
pca = joblib.load("Website_PCA_model.pkl")
print("PCA model loaded successfully.")

# Prompt for directory path
directory_path = input("Enter the path to the directory containing images: ")

if not os.path.isdir(directory_path):
    print("The specified path is not a valid directory.")
else:
    results = []
    marked_images_dir = os.path.join(directory_path, "marked_images")
    os.makedirs(marked_images_dir, exist_ok=True)

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            file_path = os.path.join(root, file_name)
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping {file_name}: Not a valid image file.")
                    continue

                image_resized = cv2.resize(image, (200, 200))
                image_scaled = image_resized.astype('float32') / 255.0
                image_cnn_input = image_scaled.reshape(1, 200, 200, 1)

                prediction = cnn_model.predict(image_cnn_input).flatten()[0]
                is_website = int(prediction >= 0.5)

                print(f"File: {file_path}, Is Website: {'Yes' if is_website else 'No'}")
                results.append({"File Path": file_path, "Is Website": 'Yes' if is_website else 'No'})

                marked_image = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
                label = "Website" if is_website else "Not Website"
                color = (0, 255, 0) if is_website else (0, 0, 255)
                cv2.putText(marked_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                marked_image_path = os.path.join(marked_images_dir, f"marked_{os.path.basename(file_path)}")
                cv2.imwrite(marked_image_path, marked_image)
                print(f"Marked image saved: {marked_image_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    output_csv = os.path.join(directory_path, "website_predictions.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
