import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out1x1pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogleNet(nn.Module):
    def __init__(self, config):
        super(GoogleNet, self).__init__()
        self.image_W = config['image_W']
        self.image_H = config['image_H']
        self.image_C = config['image_C']

        self.conv1 = nn.Conv2d(self.image_C, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, config['num_classes'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class ResNet50_Model(nn.Module):
    def __init__(self, config):
        super(ResNet50_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet50(pretrained=config['load_pretrained'])
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = torch.softmax(x, dim=1)
        return x

class ResNet18_Model(nn.Module):
    def __init__(self, config):
        super(ResNet50_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet18(pretrained=config['load_pretrained'])
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = torch.softmax(x, dim=1)
        return x

class LeNet5(nn.Module):
    def __init__(self,config):
        super(LeNet5, self).__init__()
        self.image_C = config['image_C']
        self.num_classes = config['num_classes']

        #các lớp convolution
        self.conv1 = nn.Conv2d(self.image_C, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #các lớp linear
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        # lớp convolution thứ nhất
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # lớp convolution thứ hai
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)

        # flatten tensor trước khi đưa vào lớp Linear
        x = x.view(-1, 16 * 5 * 5)

        #lớp fully connected
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)

        return x


class CNN_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']=='lenet':
            self.model=LeNet5(config)
        if config['model']=='googlenet':
            self.model=GoogleNet(config)
        if config['model']=='resnet50':
            self.model=ResNet50_Model(config)
        if config['model']=='resnet18':
            self.model=ResNet18_Model(config)
        
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self,imgs,labels=None):
        if labels is not None:
            logits=self.model(imgs)
            loss = self.loss_fn(logits, labels)
            return logits,loss
        else:
            logits=self.model(imgs)
            return logits