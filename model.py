import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR

transform = transforms.Compose([
    transforms.Resize((585, 410)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = datasets.ImageFolder(root="Dataset", transform=transform)
validation_size = int(0.2 * len(data))
train_size = len(data) - validation_size
train_data, validation_size = torch.utils.data.random_split,(data, [train_size, validation_size]) 
training = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, num_workers=2)
validation = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=2)

# image, label = data[0]
# print(image.size())

class_names = [
    '0_blue', '1_blue', '2_blue', '3_blue', '4_blue', '5_blue', '6_blue', '7_blue', '8_blue', '9_blue',
    '0_green', '1_green', '2_green', '3_green', '4_green', '5_green', '6_green', '7_green', '8_green', '9_green',
    '0_red', '1_red', '2_red', '3_red', '4_red', '5_red', '6_red', '7_red', '8_red', '9_red',
    '0_yellow', '1_yellow', '2_yellow', '3_yellow', '4_yellow', '5_yellow', '6_yellow', '7_yellow', '8_yellow', '9_yellow',
    'skip_blue', 'reverse_blue', 'draw2_blue',
    'skip_green', 'reverse_green', 'draw2_green',
    'skip_red', 'reverse_red', 'draw2_red',
    'skip_yellow', 'reverse_yellow', 'draw2_yellow',
    'wild', 'wild_draw4'
]

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(32 * 144 * 101, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 54)
        self.drop2 = nn.Dropout(p=0.25)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    net = NeuralNet()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
    patience = 5
    best_accuracy = 0.0
    epochs_without_improvement = 0

    best_accuracy = 0.0

    for epoch in range(30):
        print(f"Epoch: {epoch + 1}")
        net.train()
        running_loss = 0.0

        for i, data in enumerate(training):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()        
        print(f"loss: {running_loss / len(training): .4f}")

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validation:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"validation acc: {accuracy: .2f}%")

        scheduler.step(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            torch.save(net.state_dict(), "UNO_CNN.pth")
            print(f"Model saved with accuracy: {accuracy:.2f}%")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered. No improvement in {patience} epochs.")
            break

    print("completed")