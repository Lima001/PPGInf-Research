import sys

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from albumentations import *
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from dataset import CustomDataset

EPOCHS = 20
BATCH_SIZE = 32

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print("Configuration started...")
transform = Compose([
    Resize(224,224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

trainset = CustomDataset(device, sys.argv[2], sys.argv[1], transform=transform)
valset = CustomDataset(device, sys.argv[4], sys.argv[3], transform=transform)

model = torchvision.models.vgg16(weights='DEFAULT')
#print(model)

for param in model.parameters():
    param.require_grad = False

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 2)])
model.classifier = nn.Sequential(*features)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
loss_function = nn.CrossEntropyLoss()

torch.cuda.empty_cache()

# Transfer model to GPU
model.to(device)

# Create DataLoader objects that will help create mini-batches.
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
testloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# Train the model. In PyTorch we have to implement the training loop ourselves.
print("Main loop started...")
for epochs in range(EPOCHS):
    
    model.train() # Set model in training mode.
    
    train_loss = 0.0
    train_correct = 0
    train_batches = 0

    with tqdm(trainloader, unit="batch") as tepoch:
        
        for inputs, targets in tepoch:
            tepoch.set_description(f"Train epoch {epochs}")
            
            # Move data to GPU.
            one_hot_targets = nn.functional.one_hot(targets, num_classes=2).float()
            inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
            
            #print("Targets: ", targets, one_hot_targets)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(inputs)
            loss = loss_function(outputs, one_hot_targets)

            # Accumulate metrics.
            _, indices = torch.max(outputs.data, 1)
            #print("Indices: ", indices)
            train_correct += (indices == targets).sum().item()
            train_batches +=  1
            train_loss += loss.item()
            #print("Correct: ", train_correct)
            # Backward pass and update.
            loss.backward()
            optimizer.step()

    train_loss = train_loss / train_batches
    train_acc = train_correct / len(trainset)

    # Evaluate the model on the test dataset. Identical to loop above but without
    # weight adjustment.
    model.eval() # Set model in inference mode.

    test_loss = 0.0
    test_correct = 0
    test_batches = 0
        
    with tqdm(testloader, unit="batch") as tepoch:

        for inputs, targets in tepoch:
            tepoch.set_description(f"Test epoch {epochs}")

            one_hot_targets = nn.functional.one_hot(targets, num_classes=2).float()
            inputs, targets, one_hot_targets = inputs.to(device), targets.to(device), one_hot_targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, one_hot_targets)
            _, indices = torch.max(outputs, 1)
            test_correct += (indices == targets).sum().item()
            test_batches +=  1
            test_loss += loss.item()

    test_loss = test_loss / test_batches
    test_acc = test_correct / len(valset)

    print(f'Epoch {epochs+1}/{EPOCHS} train_loss: {train_loss:.4f} - train_acc: {train_acc:0.4f} - val_loss: {test_loss:.4f} - val_acc: {test_acc:0.4f}')

    #torch.save(model.state_dict(), f'params/vgg16_10_{epochs}.pth')