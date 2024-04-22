#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from albumentations import *
from albumentations.pytorch import ToTensorV2
from PIL import Image
from math import sqrt, ceil

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

def get_vgg16():

    model = torchvision.models.vgg16(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 2)])
    model.classifier = nn.Sequential(*features)
    nn.init.xavier_uniform_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0.0)

    return model

def get_resnet18():
    
    model = torchvision.models.resnet18(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def get_inception_v3():

    model = torchvision.models.inception_v3(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.aux_logits = False
    model.AuxLogits = None
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0.0)

    return model

def get_kernels(model, model_weights, conv_layers, counter=[0]):

    model_children = list(model.children())

    for i in range(len(model_children)):
        
        if type(model_children[i]) == nn.Conv2d:
            counter[0] += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        
        else:
            get_kernels(model_children[i], model_weights, conv_layers, counter)

def visualize_kernels_v1(model, layer_idx=0, kernel_idx=0, save=None):
    model_weights = []
    conv_layers = []
    counter = [0]
    
    get_kernels(model, model_weights, conv_layers, counter)
    
    print(f"Total convolutional layers: {counter[0]}")
    
    plt.figure()
    plt.axis('off')
    plt.imshow(model_weights[layer_idx][kernel_idx][0, :, :].detach(), cmap='gray')        
        
    if save is not None:
        plt.savefig(f'{save}/kernels_{layer_idx}.png')
    
    plt.show()
    plt.close()
        
def visualize_kernels_v2(model, layer_idx=0, save=None):
    
    model_weights = []
    conv_layers = []
    counter = [0]
    
    get_kernels(model, model_weights, conv_layers, counter)
    
    print(f"Total convolutional layers: {counter[0]}")

    grid_dim = ceil(sqrt(conv_layers[layer_idx].out_channels))
    
    plt.figure()
        
    for i, kernel in enumerate(model_weights[layer_idx]):
        plt.subplot(grid_dim, grid_dim, i+1)
        plt.imshow(kernel[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        
    if save is not None:
        plt.savefig(f'{save}/kernels_{layer_idx}.png')
    
    plt.show()
    plt.close()
    
    plt.figure()
    plt.subplot(1, 1, 1)
    processed_kernels = torch.sum(model_weights[layer_idx][0, :, :, :].data ,0) / model_weights[layer_idx][0, :, :, :].data.shape[0]

    plt.imshow(processed_kernels)
    
    if save:
        plt.savefig(f'{save}/joint_kernels_{layer_idx}.png')
        
    plt.show()
    plt.close()

def visualize_feature_maps_v1(model, image, layer_idx=0, kernel_idx=0, save=None):
    
    model_weights = []
    conv_layers = []
    counter = [0]
    
    get_kernels(model, model_weights, conv_layers, counter)
    #print(len(conv_layers))

    results = [conv_layers[0](image)]
    for i in range(1, layer_idx+1):
        results.append(conv_layers[i](results[-1]))
    
    data = results[layer_idx][0, :, :, :].data[kernel_idx]
    
    print(f"predicted: {torch.max(model(image), 1)[1]}")

    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.axis("off")
    
    if save is not None:
        plt.savefig(f'{save}/feature_maps_{layer_idx}.png')
    
    plt.show()
    plt.close()

def visualize_feature_maps_v2(model, image, layer_idx=0, save=None):
    
    model_weights = []
    conv_layers = []
    counter = [0]
    
    get_kernels(model, model_weights, conv_layers, counter)
    #print(len(conv_layers))

    results = [conv_layers[0](image)]
    for i in range(1, layer_idx+1):
        results.append(conv_layers[i](results[-1]))
    
    data = results[layer_idx][0, :, :, :].data
    
    print(f"predicted: {torch.max(model(image), 1)[1]}")

    plt.figure()
    dim = ceil(sqrt(data.size(0)))  
    
    for i, feature_map in enumerate(data):
        plt.subplot(dim, dim, i+1)
        plt.imshow(feature_map, cmap='gray')
        plt.axis("off")
    
    if save is not None:
        plt.savefig(f'{save}/feature_maps_{layer_idx}.png')
    
    plt.show()
    plt.close()
    
    plt.figure()
    plt.subplot(1, 1, 1)
    processed_features = torch.sum(data,0) / data.shape[0]
    plt.imshow(processed_features)
    
    if save is not None:
        plt.savefig(f'{save}/joint_feature_maps_{layer_idx}.png')
    
    plt.show()
    plt.close()

# Adapted from: https://github.com/justinbellucci/cnn-visualizations-pytorch/tree/master
def visualize_activation_maximization(model, epochs, layer_idx, kernel_idx, save=None):
    
    grad = {}
        
    def get_grad(name):
        def hook(model, grad_in, grad_out):
            grad[name] = grad_out[0,kernel_idx]
        return hook

    model.features[layer_idx].register_forward_hook(get_grad("conv"))    

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transform = Compose(
        [
            Resize(resize_dim,resize_dim),
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    )
    
    noisy_img = np.random.randint(125, 190, (224, 224, 3), dtype='uint8')
    transformed_img = data_transform(image=noisy_img)['image'].unsqueeze(0).requires_grad_()
    
    optimizer = torch.optim.Adam([transformed_img], lr=0.1, weight_decay=1e-6)
    
    for epoch in range(1, epochs+1):
        x = transformed_img
        optimizer.zero_grad()
        
        for idx, layer in enumerate(model.features):
                x = layer(x)
                if idx == layer_idx:
                    break

        loss = -torch.mean(grad["conv"])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            
            print(f'Epoch {epoch}/{epochs} - loss: {loss.data.numpy():.3f}')
            
            np_image = transformed_img.detach().numpy()
            np_image = np_image.squeeze(0)
            np_image = np_image.transpose(1, 2, 0)
            image = np.clip((np_image * std + mean), 0, 1)
            
            plt.figure()
            plt.imshow(image) 
            
            if save is not None:
                plt.savefig(f'{save}/activation_max_l{layer_idx}f{kernel_idx}.png')
            
            plt.show()
            plt.close()
            

# Adaptade from: https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
def visualize_saliency_map(model, img, save=None):

    img.requires_grad_()
    output = model(img)

    max_logit_index = output.argmax()
    max_logit = output[0][max_logit_index]
    
    max_logit.backward()
    saliency, _ = torch.max(img.grad.data.abs(), dim=1)
    
    plt.imshow(saliency[0])
    plt.axis('off')
    
    if save is not None:
        plt.savefig(f'{save}/saliency_map.png')
    
    plt.show()
    plt.close()

if __name__ == "__main__":

    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--image")
    parser.add_argument("--model")
    parser.add_argument("--layer_idx", type=int)
    parser.add_argument("--kernel_idx", type=int)
    parser.add_argument("--save")
    parser.add_argument("--config")
    args = parser.parse_args()
    
    #device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    
    resize_dim = 224
    if args.model == "inception_v3":
        resize_dim = 299
    
    data_transform = Compose(
        [
            Resize(resize_dim,resize_dim),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    )
    
    if args.model == "vgg16":
        model = get_vgg16()
    elif args.model == "resnet18":
        model = get_resnet18()
    elif args.model == "inception_v3":
        model = get_inception_v3()

    model.load_state_dict(torch.load(args.config, map_location=device))
    print(model)
    model.eval()
    
    img = np.array(Image.open(args.image))
    
    plt.imshow(img)
    plt.show()
    
    img = data_transform(image=img)['image']
    img = img.unsqueeze(0)

    # Example calls
    visualize_kernels_v1(model, args.layer_idx, args.kernel_idx, args.save)
    visualize_feature_maps_v1(model, img, args.layer_idx, args.kernel_idx, args.save)
    visualize_kernels_v2(model, args.layer_idx, args.save)
    visualize_feature_maps_v2(model, img, args.layer_idx, args.save)
    visualize_activation_maximization(model, 50, args.layer_idx, args.kernel_idx, args.save)
    visualize_saliency_map(model, img, args.save)
