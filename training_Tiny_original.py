import os
import torch
import sys
sys.path.append('/home/marco/Documents/Projects/MachineUnlearning/training_cifar')
from vit import ViT
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18,ResNet18_Weights, resnet34, resnet50
import tqdm
from torch.optim.lr_scheduler import _LRScheduler

from kornia.augmentation import RandomMixUpV2

def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(self, dropout_prob=0.1, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(p=dropout_prob) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(p=dropout_prob) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(384,num_classes)
        )


    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# Define data preprocessing transforms
transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Define the dataset and data loader
train_dataset = datasets.ImageFolder('train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=4)

test_dataset = datasets.ImageFolder('val/images', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=96, shuffle=True, num_workers=4)

# Initialize your model
model = resnet50(pretrained=True).to('cuda')
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).to('cuda')
model.maxpool = nn.Identity()
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, 200)).to('cuda')
#model = AllCNN(n_channels=3, num_classes=200).to('cuda')
#model = ViT(image_size=64, patch_size=4, num_classes=200, dim=512, depth=8, heads=12, mlp_dim=512, pool = 'cls', channels = 3, dim_head = 128, dropout = 0.1, emb_dropout = 0.1).to('cuda')
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1) #learning rate decay
iter_per_epoch = len(train_loader)
# Training loop
num_epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # Print the average loss for this epoch
    average_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    train_scheduler.step(val_acc)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {average_loss:.4f} Accuracy TRAIN: {train_acc:.2f}% Accuracy VAL: {val_acc:.2f}% Accuracy BEST: {best_acc:.2f}%")
    print(f"Accuracy on test set: {(val_acc):.2f}%")
    if val_acc > best_acc:
        best_acc = val_acc
    torch.save(model.state_dict(), '/home/marco/Documents/Projects/MachineUnlearning/tiny-imagenet-200/chks_tiny/best_model_tiny_ViT.pth')
    
