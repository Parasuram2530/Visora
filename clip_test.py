from clip_utils import extract_features

image_folder = "raw-img"

features, labels, class_names = extract_features(image_folder)

print(f"Feature shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"classes: {class_names}")