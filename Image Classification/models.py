import torchvision
import torch.nn as nn

import spp

class CustomConv(nn.Module):
    
    def __init__(self, num_classes):
        super(CustomConv, self).__init__()
        self.relu = nn.ReLU()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.spp_layer = spp.SpatialPyramidPooling(levels=(3,2,1))
        self.fc = nn.Linear(896, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        out = self.spp_layer(out)
        out = self.fc(out)
        
        return out

def get_custom(n_class):
    
    model = CustomConv(n_class)
    nn.init.xavier_uniform_(model.conv_layer1.weight)
    nn.init.constant_(model.conv_layer1.bias, 0.0)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)
    
    return model

def get_vgg16(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.vgg16(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.vgg16()

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, n_class)])
    model.classifier = nn.Sequential(*features)
    nn.init.xavier_uniform_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0.0)

    return model

def get_resnet18(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.resnet18(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.resnet18()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def get_resnet34(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.resnet34(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.resnet34()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def get_efficientnet_v2(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.efficientnet_v2_s(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.efficientnet_v2_s()

    model.classifier[1] = nn.Linear(1280, n_class) 
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.constant_(model.classifier[1].bias, 0.0)

    return model

def get_mobilenet_v3(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.mobilenet_v3_small()

    model.classifier[3] = nn.Linear(1024,n_class) 
    nn.init.xavier_uniform_(model.classifier[3].weight)
    nn.init.constant_(model.classifier[3].bias, 0.0)

    return model

def get_inception_v3(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.inception_v3(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.inception_v3()

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    model.aux_logits = False
    model.AuxLogits = None
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)
    
    #num_features = model.AuxLogits.fc.in_features
    #model.AuxLogits.fc = nn.Linear(num_features, n_class)
    #nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
    #nn.init.constant_(model.AuxLogits.fc.bias, 0.0)
    
    return model

def get_vit_b16(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.vit_b_16(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.vit_b_16()

    model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=n_class))
    nn.init.xavier_uniform_(model.heads[0].weight)
    nn.init.constant_(model.heads[0].bias, 0.0)

    return model

def get_model(name, n_class, pretrained=True):
        
    if name == "vgg16":
        model = get_vgg16(n_class, pretrained)
    elif name == "resnet18":
        model = get_resnet18(n_class, pretrained)
    elif name == "resnet34":
        model = get_resnet34(n_class, pretrained)
    elif name == "inception_v3":
        model = get_inception_v3(n_class, pretrained)
    elif name == "efficientnet_v2":
        model = get_efficientnet_v2(n_class, pretrained)
    elif name == "mobilenet_v3":
        model = get_mobilenet_v3(n_class, pretrained)
    elif name == "vit_b16":
        model = get_vit_b16(n_class, pretrained)
    elif name == "custom":
        model = get_custom(n_class)
        
    return model