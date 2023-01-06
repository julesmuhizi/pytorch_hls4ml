import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchinfo import summary
import tqdm as tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # fully connected layer, output 10 classes
        self.flatten = nn.Flatten(1,3)
        self.out = nn.Linear(28 * 28, 10)
    def forward(self, x):
        # flatten the input to (batch_size, 1 * 28 * 28)
        x = self.flatten(x)     
        output = self.out(x)
        return output    

model = MLP()
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=False)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=False)

train_batch_size = 100
test_batch_size = 1

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=train_batch_size,
                 shuffle=True)

test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=test_batch_size,
                shuffle=False)

PATH = 'model/mnist_MLP.pth'
if not os.path.exists(PATH):
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in tqdm.tqdm(enumerate(train_loader, 0), total = len(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
    torch.save(PATH)
    print('Finished Training')
else:
    model = torch.load(PATH)
    print('Loaded Pretrained Model')
model.to(device)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

torch.save(model, PATH)
