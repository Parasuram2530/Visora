import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from train import CLIPCLASSIFIER

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "elephant.jpg"
input_image = Image.open(image_path).convert("RGB")
preprocessed_image = preprocess(input_image).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(preprocessed_image).float()

classifier = CLIPCLASSIFIER(input_dim=512, num_classes=10)
classifier.load_state_dict(torch.load("models/clip_classifier.pth", map_location=device))
classifier.to(device)
classifier.eval()

cam = GradCAM(classifier, target_layer="conv1")

image_features = image_features.to(device)
output = classifier(image_features)
predicted_class = output.argmax().item()

activation_map = cam(predicted_class, output)[0].detach().cpu()

activation_map_resized = transforms.Resize(preprocessed_image.shape[-2:])(activation_map.unsqueeze(0))


input_pil = to_pil_image(preprocessed_image.squeeze().cpu())
heatmap = overlay_mask(input_pil, to_pil_image(activation_map_resized.squeeze(0)), alpha=0.6)

plt.figure(figsize=(6,6))
plt.imshow(heatmap)
plt.title(f"GradCam for predicted class {predicted_class}")
plt.axis("off")
plt.tight_layout()
plt.show()