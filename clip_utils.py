import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def extract_features(image_folder):
    image_features = []
    labels = []
    class_names = sorted(os.listdir(image_folder))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(image_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in tqdm(os.listdir(class_path), desc=f"Preprocessing {class_name}"):
            img_path = os.path.join(class_path, img_name)

            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image)
                    features = features / features.norm(dim=-1, keepdim=True)
                    image_features.append(features.cpu())
                    labels.append(class_to_idx[class_name])

            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    image_features = torch.cat(image_features, dim=0)
    labels = torch.tensor(labels)

    os.makedirs("features", exist_ok=True)
    torch.save(image_features, "features/features.pt")
    torch.save(labels, "features/labels.pt")

    print("Success")
    return image_features, labels, class_names