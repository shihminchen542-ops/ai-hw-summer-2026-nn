
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =============================
# 1. Basic setup
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =============================
# 2. Load MNIST dataset
# =============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False
)


# =============================
# 3. MLP model
# =============================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# =============================
# 4. CNN model
# =============================

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# =============================
# 5. Train function
# =============================

def train_model(model, train_loader, epochs=3):
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch}: Loss = {average_loss:.4f}, Train Accuracy = {accuracy:.2f}%")

    return model


# =============================
# 6. Test function
# =============================

def test_model(model, test_loader):
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total * 100

    return accuracy, np.array(all_predictions), np.array(all_labels)


# =============================
# 7. Save sample predictions
# =============================

def save_sample_predictions(model, test_loader):
    model.eval()

    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

    plt.figure(figsize=(12, 4))

    for i in range(12):
        plt.subplot(2, 6, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Pred: {predictions[i].item()}\nTrue: {labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.close()


# =============================
# 8. Save confusion matrix
# =============================

def save_confusion_matrix(predictions, labels):
    matrix = np.zeros((10, 10), dtype=int)

    for true_label, pred_label in zip(labels, predictions):
        matrix[true_label, pred_label] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()


# =============================
# 9. Main program
# =============================

print("\nTraining MLP model")
mlp = MLP()
mlp = train_model(mlp, train_loader, epochs=3)

mlp_accuracy, mlp_predictions, mlp_labels = test_model(mlp, test_loader)
print(f"MLP Test Accuracy: {mlp_accuracy:.2f}%")


print("\nTraining CNN model")
cnn = SimpleCNN()
cnn = train_model(cnn, train_loader, epochs=3)

cnn_accuracy, cnn_predictions, cnn_labels = test_model(cnn, test_loader)
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}%")


save_sample_predictions(cnn, test_loader)
save_confusion_matrix(cnn_predictions, cnn_labels)


with open("summary.txt", "w") as f:
    f.write("MNIST Recognition by Neural Networks\n")
    f.write("====================================\n\n")
    f.write("Dataset: MNIST\n")
    f.write("Training data: torchvision.datasets.MNIST(train=True)\n")
    f.write("Testing data: torchvision.datasets.MNIST(train=False)\n\n")
    f.write("Models:\n")
    f.write("1. Shallow Multi-Layer Perceptron (MLP)\n")
    f.write("2. Convolutional Neural Network (CNN)\n\n")
    f.write("Testing Results:\n")
    f.write(f"MLP Test Accuracy: {mlp_accuracy:.2f}%\n")
    f.write(f"CNN Test Accuracy: {cnn_accuracy:.2f}%\n\n")
    f.write("Note: The models were trained only on the training set and tested on the test set.\n")


print("\nCreated files:")
print("mnist_assignment.py")
print("summary.txt")
print("sample_predictions.png")
print("confusion_matrix.png")
