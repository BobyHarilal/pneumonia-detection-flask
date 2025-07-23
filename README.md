# pneumonia-detection-flask

This project is a deep learning-based medical image classifier that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). It features a trained model deployed via a Flask-based web app where users can upload images and get real-time predictions.

## 🔍 Project Overview

- Detects pneumonia using chest X-ray images.
- Built using Python, TensorFlow/Keras, NumPy, and Flask.
- Real-time web interface for uploading and predicting.

## 🧠 Skills Applied

- Deep Learning (CNN architecture, binary classification)
- Python Programming
- Image Preprocessing (resize, normalize)
- Web Development (Flask + HTML/CSS)
- Real-time Prediction Deployment

## 📁 Folder Structure

pneumonia-detection-flask/
│
├── uiservernew.py # Flask server script (includes embedded HTML)
├── best_modelprep.py # Model training script
├── best_model.h5 # Trained CNN model file
├── static/
│ └── lungs-glowing-low-poly.jpg # Background image used in the UI

✅ You can copy this tree structure using triple backticks (\`\`\`) in markdown to format it properly.

## ⚙️ Installation

1. Clone the repository:
   git clone https://github.com/BobyHarilal/pneumonia-detection-flask.git
cd pneumonia-detection-flask

2. Install the required packages:
 pip install tensorflow flask pillow numpy

3. Ensure `best_model.h5` and the image file are placed correctly as shown in the folder structure above.

4. Run the Flask app:
   python uiservernew.py
   
## 🚀 Usage

- Open your browser and go to: http://127.0.0.1:5000
- Upload a chest X-ray image (JPG/PNG format).
- Click “Predict Pneumonia” to get the classification result (Normal or Pneumonia).

## 🧪 Model Info

- Image Size: 150x150
- Layers Used: Conv2D, MaxPooling2D, Flatten, Dense, Dropout
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Activation Functions: ReLU and Sigmoid
- Accuracy: ~96% (binary classification)

## 🖼️ Output Example

![Prediction Screenshot](static/lungs-glowing-low-poly.jpg)

## ✅ Features

- Flask-based UI with embedded HTML and CSS
- Lightweight and easy to run
- Suitable for real-time pneumonia detection

## 📌 Limitations

- Currently supports only binary classification (Normal/Pneumonia)
- Accuracy may vary depending on input quality and image resolution

## 💡 Future Enhancements

- Add Grad-CAM heatmap visualization
- Deploy on cloud platforms (e.g., Render, Heroku)
- Expand to detect other diseases (COVID, TB, etc.)


