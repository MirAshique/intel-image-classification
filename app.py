from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import os

# ─── Config ───────────────────────────────────────────
MODEL_PATH  = 'resnet50_intel.pth'
NUM_CLASSES = 6
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# ──────────────────────────────────────────────────────

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load Model Once at Startup ───────────────────────
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()
print("✅ Model loaded! Flask API ready.")

# ─── Transform ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Routes ───────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message'  : 'Intel Image Classification API',
        'model'    : 'ResNet50',
        'classes'  : CLASS_NAMES,
        'endpoint' : 'POST /predict with image file'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(tensor)
            probs  = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        # All class probabilities
        all_probs = {
            CLASS_NAMES[i]: round(probs[0][i].item() * 100, 2)
            for i in range(NUM_CLASSES)
        }

        return jsonify({
            'predicted_class': CLASS_NAMES[pred.item()],
            'confidence'     : round(conf.item() * 100, 2),
            'all_probabilities': all_probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Run ──────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)