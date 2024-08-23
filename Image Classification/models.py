# This module defines various deep learning model architectures for image classification tasks using PyTorch. 
# It includes popular models such as VGG16, ResNet18, ResNet34, Inception-V3, EfficientNet-V2, 
# MobileNet-V3, Vision Transformer (ViT B16), and a custom convolutional neural network with Spatial Pyramid Pooling. 
# The module allows for the option to load pre-trained weights (ImageNet) and customizes the final classification layers 
# based on the number of classes specified by the user.

import torchvision
import torch.nn as nn
import spp

# Define a custom shallow convolutional neural network with Spatial Pyramid Pooling (SPP).
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

# Initialize and return the custom model with Xavier uniform initialization
def get_custom(n_class):
    
    model = CustomConv(n_class)
    nn.init.xavier_uniform_(model.conv_layer1.weight)
    nn.init.constant_(model.conv_layer1.bias, 0.0)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)
    
    return model

# Initialize and return a modified VGG16 model
def get_vgg16(n_class, pretrained=True):

    if pretrained:    
        # Load VGG16 with pre-trained weights from ImageNet
        model = torchvision.models.vgg16(weights="DEFAULT") 

        # Freeze the parameters to prevent updates during training
        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.vgg16()

    # Replace the final classifier layer to match the number of classes
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, n_class)])
    model.classifier = nn.Sequential(*features)
    nn.init.xavier_uniform_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0.0)

    return model

# Initialize and return a modified ResNet18 model
def get_resnet18(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.resnet18(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.resnet18()

    # Replace the final classifier layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

# Initialize and return a modified ResNet34 model
def get_resnet34(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.resnet34(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.resnet34()

    # Replace the final classifier layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

# Initialize and return a modified EfficientNet-V2 model
def get_efficientnet_v2(n_class, pretrained=True):
    
    if pretrained:    
        
        model = torchvision.models.efficientnet_v2_s(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.efficientnet_v2_s()

    # Replace the final classifier layer to match the number of classes
    model.classifier[1] = nn.Linear(1280, n_class) 
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.constant_(model.classifier[1].bias, 0.0)

    return model

# Initialize and return a modified MobileNet-V3 model
def get_mobilenet_v3(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.mobilenet_v3_small()

    # Replace the final classifier layer to match the number of classes
    model.classifier[3] = nn.Linear(1024,n_class) 
    nn.init.xavier_uniform_(model.classifier[3].weight)
    nn.init.constant_(model.classifier[3].bias, 0.0)

    return model

# Initialize and return a modified Inception-V3 model
def get_inception_v3(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.inception_v3(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.inception_v3()

    # Replace the final classifier layer to match the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, n_class)
    
    # Define to not use auxiliar logits
    model.aux_logits = False
    model.AuxLogits = None
    
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)
    
    # Uncomment to use auxiliar logits
    #num_features = model.AuxLogits.fc.in_features
    #model.AuxLogits.fc = nn.Linear(num_features, n_class)
    #nn.init.xavier_uniform_(model.AuxLogits.fc.weight)
    #nn.init.constant_(model.AuxLogits.fc.bias, 0.0)
    
    return model

# Initialize and return a modified Vision Transformer (B16) model
def get_vit_b16(n_class, pretrained=True):

    if pretrained:    
        
        model = torchvision.models.vit_b_16(weights="DEFAULT")

        for param in model.parameters():
            param.requires_grad = False
    
    else:
        model = torchvision.models.vit_b_16()

    # Replace the final classifier layer to match the number of classes
    model.heads = nn.Sequential(nn.Linear(in_features=768, out_features=n_class))
    nn.init.xavier_uniform_(model.heads[0].weight)
    nn.init.constant_(model.heads[0].bias, 0.0)

    return model

# Function to select and return the appropriate model based on the model name
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