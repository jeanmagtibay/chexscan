from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import torch
from torchvision.models import resnet18

app = Flask(__name__)

# Load the model architecture
model = resnet18(pretrained=True)
num_classes = 3  # Assuming you have 3 classes (Normal, Pneumonia, Tuberculosis)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a function to perform predictions
def predict(image_path):
    image = Image.open(image_path)
    image = test_transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = 'uploaded_image.jpg'  # Save the uploaded image temporarily
        file.save(image_path)
        predicted_class = predict(image_path)
        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
