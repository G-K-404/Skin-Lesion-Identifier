import os
import base64
from flask import Flask, render_template, request
from io import BytesIO
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import models, transforms

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

skin_lesions = ["BA- cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus", "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"]

num_classes = 8
model = models.densenet121(pretrained=False)
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(torch.load(r"models\densenet121_trained_model.pth", map_location=device))
model = model.to(device)
model.eval()

def predict_image(image):
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item()
    return predicted_idx, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        image = Image.open(file).convert('RGB')
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        image_data = f"data:image/png;base64,{encoded_image}"
        
        predicted_idx, confidence = predict_image(image)
        
        if confidence >= 0.60:
            result = f"Probable Lesion: {skin_lesions[predicted_idx]}, Confidence: {confidence:.4f}"
        else:
            result = "No lesion detected"
            
        return render_template('index.html', prediction=result, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
