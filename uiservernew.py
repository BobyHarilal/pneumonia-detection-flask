from flask import Flask, request, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = load_model("best_model.h5")

# Ensure the "uploads" folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define function to process and predict images
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "The person has Pneumonia." if prediction[0][0] > 0.5 else "The person does NOT have Pneumonia."

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Medical Diagnosis</title>
    <style>
        body {
            background-image: url('static/lungs-glowing-low-poly.jpg');
            background-size: cover;
            background-position: center;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .upload-container {
            background-color: rgba(255, 255, 255, 0.92);
             width: 50%;
            margin: auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);
        }
        h1 {
            font-size: 24px;
            margin-bottom: 20px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 20px;
        }
        button {
            background-color: #0062cc;
            color: #fff;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
            color: #ff0000;
        }
    </style>
</head>
<body>
    <div class="upload-container">
        <h2>Upload X-ray Image for Diagnosis</h2>
        
        <form action="/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit">Upload and Diagnose</button>
        </form>

        {% if result %}
            <div class="result">{{ result }}</div>
        {% endif %}

        {% if image_path %}
            <h3>Uploaded Image:</h3>
            <img src="{{ image_path }}" alt="Uploaded X-ray">
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template_string(HTML_TEMPLATE, result="No file uploaded.", image_path=None)

        file = request.files["file"]
        if file.filename == "":
            return render_template_string(HTML_TEMPLATE, result="No selected file.", image_path=None)

        # Save file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict result
        result = predict_image(file_path)

        return render_template_string(HTML_TEMPLATE, result=result, image_path=file_path)

    return render_template_string(HTML_TEMPLATE, result=result, image_path=image_path)


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    with open('static/lungs-glowing-low-poly.jpg', 'wb') as f:
        f.write(open('lungs-glowing-low-poly.jpg', 'rb').read())
    app.run(debug=True)

