import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50_Model(nn.Module):
    def __init__(self, config):
        super(ResNet50_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet50(pretrained=config['load_pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x

class ResNet18_Model(nn.Module):
    def __init__(self, config):
        super(ResNet18_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.resnet18(pretrained=config['load_pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x

class VGG16_Model(nn.Module):
    def __init__(self, config):
        super(VGG16_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.vgg16(pretrained=config['load_pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.classifier[-1] = nn.Linear(self.cnn.classifier[-1].in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x
    
class VGG19_Model(nn.Module):
    def __init__(self, config):
        super(VGG19_Model, self).__init__()
        self.num_classes = config['num_classes']
        self.cnn = models.vgg19(pretrained=config['load_pretrained'])
        if config['freeze']:
            for param in self.cnn.parameters():
                param.requires_grad=False
        self.cnn.classifier[-1] = nn.Linear(self.cnn.classifier[-1].in_features, self.num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = F.softmax(x, dim=-1)
        return x

class CNN_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['model']=='resnet50':
            self.model=ResNet50_Model(config)
        if config['model']=='resnet18':
            self.model=ResNet18_Model(config)
        if config['model']=='vgg16':
            self.model=VGG16_Model(config)
        if config['model']=='vgg19':
            self.model=VGG19_Model(config)
        
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(self,imgs,labels=None):
        if labels is not None:
            logits=self.model(imgs)
            loss = self.loss_fn(logits, labels)
            return logits,loss
        else:
            logits=self.model(imgs)
            return logits