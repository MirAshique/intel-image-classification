import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

# ─── Config ───────────────────────────────────────────
MODEL_PATH  = 'resnet50_intel.pth'
TEST_DIR    = 'dataset/seg_test/seg_test'
NUM_CLASSES = 6
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# ──────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Load Model ───────────────────────────────────────
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
    print("✅ Model loaded!")
    return model

# ─── Transform ────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Single Image Inference ───────────────────────────
def predict_single(model, image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = inference_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return {
        'class': CLASS_NAMES[pred.item()],
        'confidence': round(conf.item() * 100, 2)
    }

# ─── Batch Inference (4 samples) ─────────────────────
def predict_batch(model, test_dir, num_samples=4):
    images, labels, paths = [], [], []
    for cls in sorted(os.listdir(test_dir))[:num_samples]:
        cls_path = os.path.join(test_dir, cls)
        img_file = os.listdir(cls_path)[0]
        full_path = os.path.join(cls_path, img_file)
        paths.append(full_path)
        labels.append(cls)
        img = Image.open(full_path).convert('RGB')
        images.append(inference_transform(img))

    batch = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs   = F.softmax(outputs, dim=1)
        confs, preds = torch.max(probs, 1)

    print("\n" + "=" * 50)
    print("   BATCH INFERENCE RESULTS (4 Samples)")
    print("=" * 50)
    for i in range(num_samples):
        actual     = labels[i]
        predicted  = CLASS_NAMES[preds[i].item()]
        confidence = confs[i].item() * 100
        status     = "CORRECT" if actual == predicted else "WRONG"
        print(f"Sample {i+1}:")
        print(f"  Actual     : {actual}")
        print(f"  Predicted  : {predicted} [{status}]")
        print(f"  Confidence : {confidence:.2f}%")
        print("-" * 50)

# ─── Main ─────────────────────────────────────────────
if __name__ == '__main__':
    model = load_model()
    predict_batch(model, TEST_DIR)