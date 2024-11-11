from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
from PIL import Image
import json

app = Flask(__name__)

# Load your YOLOv8 model
model = YOLO(r'output/train5/weights/best.pt')

# Load the label to Tamil mapping
# Update the path to where your JSON file is located
with open(r'C:\Users\laksh\Documents\sample\label_to_tamil.json', 'r', encoding='utf-8') as f:
    label_to_tamil = json.load(f)  #for reading the data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    img = Image.open(file.stream)  #Opens the uploaded image using Pillow (PIL)
    
    # Perform inference
    results = model(img)  #the loaded image is passed into the yolo model
    
    # Extract labels and boxes
    output = []
    for r in results[0].boxes.data.tolist():
        x_min, y_min, x_max, y_max, confidence, class_id = r
        label = results[0].names[int(class_id)]  #to get the class names which is mapped with the help of the class id
        tamil_letter = label_to_tamil.get(label, "Unknown")  # Get the Tamil equivalent
        output.append({
            'label': label,
            'tamil_letter': tamil_letter,
            'box': [x_min, y_min, x_max, y_max],
            'confidence': confidence
        })

    return jsonify({'predictions': output})

if __name__ == '__main__':
    app.run(debug=True)
