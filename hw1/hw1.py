# Import the required modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

# Fix the randomness
seed = 1234
torch.manual_seed(seed)

# Download the dataset, and split it into Train, Val, and Test sets.
transform = T.Compose([
    T.ToTensor(),
    T.Grayscale(),
    T.Normalize(mean=(0.5,), std=(0.5,))
])

train_set = CIFAR10(root='CIFAR10', train=True, 
                    transform=transform, download=True)

train_set_length = int(0.8 * len(train_set))
val_set_length = len(train_set) - train_set_length

train_set, val_set = random_split(train_set, [train_set_length, val_set_length])

test_set = CIFAR10(root='CIFAR10', train=False, 
                    transform=transform, download=True)

# Define the data loaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# ANN with 0 hidden layer
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = self.layer1(x)
      return x
  
#ANN with 1 hidden layer  
class MyModelRelu(nn.Module):
    def __init__(self, num_layer1):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.relu(self.layer1(x))
      x = self.layer2(x)
      return x
class MyModelSigmoid(nn.Module):
    def __init__(self, num_layer1):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.sigmoid(self.layer1(x))
      x = self.layer2(x)
      return x
class MyModelTanh(nn.Module):
    def __init__(self, num_layer1):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.tanh(self.layer1(x))
      x = self.layer2(x)
      return x

#ANN with 2 hidden layers  
class MyModel2Relu(nn.Module):
    def __init__(self, num_layer1, num_layer2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      x = self.layer3(x)
      return x
class MyModelReluSigmoid(nn.Module):
    def __init__(self, num_layer1, num_layer2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.relu(self.layer1(x))
      x = F.sigmoid(self.layer2(x))
      x = self.layer3(x)
      return x
class MyModelReluTanh(nn.Module):
    def __init__(self, num_layer1, num_layer2):
        super().__init__()
        self.layer1 = nn.Linear(in_features=32*32, out_features=num_layer1)
        self.layer2 = nn.Linear(in_features=num_layer1, out_features=num_layer2)
        self.layer3 = nn.Linear(in_features=num_layer2, out_features=10)
    def forward(self, x):
      x = torch.flatten(x, 1)
      x = F.relu(self.layer1(x))
      x = F.tanh(self.layer2(x))
      x = self.layer3(x)
      return x

# logits --> unnormalized probabilites
#nn.BatchNorm1d
#nn.Dropout

# Instantiate the model and Train it for 3 epochs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

file = open('results.txt', 'w')

#1-Layer network
file.write("1-Layer Network\n")
zeroLayer=[1e-4,1e-5]
for m in range(2):
    file.write(str(zeroLayer[m])+"\n")
    model = MyModel().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=zeroLayer[m])

    num_epochs = 20
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        accum_train_loss = 0
        for i, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = loss_function(output, labels)

            # accumlate the loss
            accum_train_loss += loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        accum_val_loss = 0
        with torch.no_grad():
            for j, (imgs, labels) in enumerate(val_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                accum_val_loss += loss_function(output, labels).item()

        # print statistics of the epoch
        file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
        print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
        
    # Compute Test Accuracy
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            
            _, predicted_labels = torch.max(output, 1)
            correct += (predicted_labels == labels).sum()
            total += labels.size(0)
    file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
    print(f'Test Accuracy = {100 * correct/total :.3f}%')
  
#2-Layer Network with relu
file.write("2-Layer Network with ReLU\n")
hiddenUnits=[512, 256, 128]
learningRates=[1e-4, 1e-5]
for m in range(3):
    for n in range(2):
        file.write(str(hiddenUnits[m])+"-"+str(learningRates[n])+"\n")
        model = MyModelRelu(hiddenUnits[m]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')
        
#2-Layer Network with sigmoid
file.write("2-Layer Network with Sigmoid\n")
for m in range(3):
    for n in range(2):
        file.write(str(hiddenUnits[m])+"-"+str(learningRates[n])+"\n")
        model = MyModelSigmoid(hiddenUnits[m]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')
        
#2-Layer Network with tanh
file.write("2-Layer Network with Tanh\n")
for m in range(3):
    for n in range(2):
        file.write(str(hiddenUnits[m])+"-"+str(learningRates[n])+"\n")
        model = MyModelTanh(hiddenUnits[m]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')

#3-Layer Network with relu-sigmoid
file.write("3-Layer Network with ReLU-Sigmoid\n")
hiddenUnits=[(512,256), (512,128), (256,128), (256,64), (128,64)]
learningRates=[1e-4, 1e-5]
for m in range(5):
    for n in range(2):
        file.write(str(hiddenUnits[m][0])+"-"+str(hiddenUnits[m][1])+"-"+str(learningRates[n])+"\n")
        model = MyModelReluSigmoid(hiddenUnits[m][0],hiddenUnits[m][1]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')
        
#3-Layer Network with relu-tanh
file.write("3-Layer Network with ReLU-Tanh\n")
for m in range(5):
    for n in range(2):
        file.write(str(hiddenUnits[m][0])+"-"+str(hiddenUnits[m][1])+"-"+str(learningRates[n])+"\n")
        model = MyModelReluTanh(hiddenUnits[m][0],hiddenUnits[m][1]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')
        
#3-Layer Network with relu-relu
file.write("3-Layer Network with ReLU-ReLU\n")
for m in range(5):
    for n in range(2):
        file.write(str(hiddenUnits[m][0])+"-"+str(hiddenUnits[m][1])+"-"+str(learningRates[n])+"\n")
        model = MyModel2Relu(hiddenUnits[m][0],hiddenUnits[m][1]).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRates[n])

        num_epochs = 20
        for epoch in tqdm(range(num_epochs)):
            # Training
            model.train()
            accum_train_loss = 0
            for i, (imgs, labels) in enumerate(train_loader, start=1):
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                loss = loss_function(output, labels)

                # accumlate the loss
                accum_train_loss += loss.item()

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            accum_val_loss = 0
            with torch.no_grad():
                for j, (imgs, labels) in enumerate(val_loader, start=1):
                    imgs, labels = imgs.to(device), labels.to(device)
                    output = model(imgs)
                    accum_val_loss += loss_function(output, labels).item()

            # print statistics of the epoch
            file.write(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}\n')
            print(f'Epoch = {epoch} | Train Loss = {accum_train_loss / i:.4f}\tVal Loss = {accum_val_loss / j:.4f}')
            
        # Compute Test Accuracy
        model.eval()
        with torch.no_grad():
            correct = total = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                
                _, predicted_labels = torch.max(output, 1)
                correct += (predicted_labels == labels).sum()
                total += labels.size(0)
        file.write(f'Test Accuracy = {100 * correct/total :.3f}%\n')
        print(f'Test Accuracy = {100 * correct/total :.3f}%')
file.close()