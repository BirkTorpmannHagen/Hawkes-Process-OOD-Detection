from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet_snn import SResnet

def store_spike_trains(spike_trains, name, path_to_folder):
    torch.save(spike_trains, path_to_folder/ f'{name}.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.__version__)
print(torch.cuda.is_available())

# Your SResnet class and SurrogateBPFunction go here

# MNIST Data loading
transform = transforms.Compose([
    transforms.Resize((16, 16)),  # Resize the images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../../Datasets/MNIST', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='../../Datasets/MNIST', train=False, transform=transform, download=True)
ood_test_dataset = datasets.EMNIST(root='../../Datasets/emnist', split="letters", train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
ood_test_loader = DataLoader(dataset=ood_test_dataset, batch_size=64, shuffle=False)

# Model initialization
n = 1 # Number of blocks per layer, adjust as needed
nFilters = 8 # Number of filters, adjust as needed
num_steps = 10 # Number of time steps, adjust as needed

model = SResnet(n=n, nFilters=nFilters, num_steps=num_steps, img_size=16, num_cls=10)
model.cuda()  # Move the model to GPU

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, spike_trains = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def ood_testing(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(ood_test_loader):
            data, target = data.to(device), target.to(device)
            print(target)
            output, spike_trains = model(data)
            spike_trains["labels"] = target
            store_spike_trains(spike_trains, f"batch_{idx}", Path(__file__).parent / "spike_trains" / "convolutional_spike_trains")
            #test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\n OOD Test set, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, 3):  # 10 epochs, adjust as needed
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        ood_testing(model, device, test_loader)