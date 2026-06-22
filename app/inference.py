import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.models.model_factory import get_model



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Load models once
@torch.no_grad()
def load_models():
    models = []
    names = ["mobilenet", "efficientnet", "resnet"]

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for name in names:
        model = get_model(name, num_classes=3, freeze=False)

        model_path = os.path.join(BASE_DIR, "saved_models", f"{name}.pt")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        models.append(model)

    return models

models = load_models()

# Prediction
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    probs_list = []

    for model in models:
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        probs_list.append(probs)

    probs_stack = torch.stack(probs_list)

    # Weighted voting
    weights = torch.tensor([0.80, 0.85, 0.95]).to(device)
    weights = weights / weights.sum()

    final_probs = torch.zeros_like(probs_stack[0])

    for i in range(len(weights)):
        final_probs += weights[i] * probs_stack[i]

    pred = torch.argmax(final_probs, dim=1).item()
    confidence = torch.max(final_probs).item()

    return pred, confidence