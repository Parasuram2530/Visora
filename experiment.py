import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
 
# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
# Load and preprocess image
image_path = "cat.jpeg"  # Replace with your own image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
 
# Candidate text descriptions
text_descriptions = [
    "A photo of a cat",
    "A photo of a dog",
    "A bowl of food",
    "A sunny beach",
    "A person riding a bike"
]
 
# Tokenize text inputs
text_tokens = clip.tokenize(text_descriptions).to(device)
 
# Encode image and text with CLIP
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)
 
    # Compute cosine similarity
    logits_per_image, _ = model(image, text_tokens)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
 
# Display results
plt.imshow(Image.open(image_path))
plt.axis("off")
plt.title("📷 Input Image")
plt.show()
 
# Print matched results
print("🧠 Top Text Matches for Image:")
for i, (desc, prob) in enumerate(zip(text_descriptions, probs[0])):
    print(f"{i+1}. {desc} - {prob:.2%}")