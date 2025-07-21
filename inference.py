# inference_utils.py
import torch
import joblib
import clip
from PIL import Image
from train import CLIPCLASSIFIER
import json

with open("class_mapping.json", "r") as f:
    CLASS_NAMES = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def load_classifier():
    model = CLIPCLASSIFIER(512, 10)  
    state_dict = torch.load("models/clip_classifier.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def load_scaler():
    return joblib.load("models/clip_scalar.pkl")

def predict(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).float()

    scaler = load_scaler()
    image_features = scaler.transform(image_features.cpu().numpy())
    image_features = torch.tensor(image_features, dtype=torch.float32).to(device)

    classifier = load_classifier()
    with torch.no_grad():
        outputs = classifier(image_features)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    class_name = CLASS_NAMES.get(str(predicted_class).capitalize(), "Unknown")
    return class_name
