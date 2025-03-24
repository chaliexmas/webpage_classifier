# WebPageClassifier Using CNN

This repository contains scripts for classifying images as websites using a trained Convolutional Neural Network (CNN). The project includes two scripts:

1. **WebRecognizer_Create_Model.py**: Trains a CNN model to recognize websites.
2. **WebRecognizer_Evaluater.py**: Loads a trained model and classifies images in a given directory.

## Features
- Uses a CNN to classify images as websites.
- Implements Principal Component Analysis (PCA) for dimensionality reduction.
- Automatically searches a directory and all subdirectories for PNG and JPG images.
- Outputs results as a CSV file with full paths to identified website images.
- Generates marked images with classification labels.

---

## Setup Instructions
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy opencv-python pandas matplotlib scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/webpageclassifier.git
cd webpageclassifier
```

---

## Training the Model
The training script preprocesses images, applies PCA, and trains a CNN model.

### Run Training
```bash
python WebRecognizer_Create_Model.py
```
- Expects a dataset structured as:
  ```
  datasets/
    ├── websites/       # Images labeled as websites
  ```
- The trained model will be saved as `Website_CNN_model.h5`.
- PCA transformation will be stored as `Website_PCA_model.pkl`.

---

## Running Inference
The inference script loads the trained model and classifies images in a specified directory.

### Run Inference
```bash
python WebRecognizer_Evaluater.py
```
- Prompts for a directory path.
- Searches all subdirectories for PNG and JPG images.
- Saves predictions in `website_predictions.csv`.
- Marks images with classification labels (`Website` / `Not Website`).

---

## Output
- **CSV File:** List of identified website images with full file paths.
- **Marked Images:** Saved in `marked_images/` inside the input directory.

### Example CSV Output:
```
File Path,Is Website
/path/to/image1.png,Yes
/path/to/image2.jpg,No
```

---

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

---

## License
This project is licensed under the MIT License.

