import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

features = torch.load("features/features.pt", weights_only=True)
features = features.float()
labels = torch.load("features/labels.pt", weights_only=True)
# torch.load("features/features.pt", weights_only=True)  # if saved with weights only

# print("Features")
# print(features)
# print("Labels")
# print(labels)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
features = scalar.fit_transform(features.cpu().numpy())
features = torch.tensor(features, dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)

class CLIPCLASSIFIER(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)
    

if __name__ == "__main__":
    
    num_classes = len(labels.unique())
    model = CLIPCLASSIFIER(512, num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.float().to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    epochs = 15

    for epoch in range(epochs):
        model.train()
        inputs = X_train.to(device).float()
        targets = y_train.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device).float())
            val_preds = val_outputs.argmax(dim=1).cpu()
            acc = accuracy_score(y_val, val_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss - {loss.item():4f} - Val Acc: {acc:4f}")


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/clip_classifier.pth")
    print("Model Saved successful.")

    joblib.dump(scalar, "models/clip_scalar.pkl")
    print(classification_report(y_val, val_preds))