from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import sys
import os
import sys
import os

# 1. 모델 구조 정의 (User Provided Architecture)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 특징 추출부 (Feature Extractor)
        self.conv = nn.Sequential(
            # Block 1: 48x48 -> 24x24
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 분류기 (Classifier)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256), # Fully Connected 층에도 BN 적용 가능
            nn.Dropout(p=0.4),    # Dropout 비율을 살짝 높여 과적합 방지
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

from flask_cors import CORS

app = Flask(__name__, template_folder='.')
CORS(app) # Enable CORS for all routes

# 2. 모델 로드 및 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

num_classes = 10 # animals10 기준
model = CNN(num_classes=num_classes).to(device)

model_path = 'animals10_cnn.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {model_path} not found.")

model.eval()

# 클래스 이름 가져오기
import translate
classes = [translate.translate.get(c, c) for c in ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']]

# 3. 이미지 전처리 함수
def transform_image(image_bytes):
    # User's model uses 48x48 input based on the architecture comment "Block 1: 48x48 -> 24x24"
    my_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0).to(device)

@app.route('/')
def index():
    return render_template('index.html', classes=classes)

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
        
        result = classes[predicted.item()]
        return jsonify({'animal': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)