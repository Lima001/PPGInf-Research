import matplotlib
matplotlib.use('tkagg')     # This line is necessary on my local enviorment; you may choose a different backend for matplot

import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from albumentations import *
from albumentations.pytorch import ToTensorV2
from PIL import Image
from functools import reduce
from torchvision.models.feature_extraction import create_feature_extractor

from models import *
from transform import *

def visualize_kernel(model, module_name, kernel_idx=0, save=None):

    names = module_name.split(sep='.')
    module = reduce(getattr, names, model)    

    plt.figure()
    plt.axis('off')
    plt.imshow(module.weight[kernel_idx][0, :, :].detach())        
        
    if save is not None:
        plt.savefig(f'{save}/kernels_{module_name}_k{kernel_idx}.png')
    else:
        plt.show()
    
    plt.close()
        

def visualize_feature_map(model, image, module_name, kernel_idx=0, save=None):

    extractor = create_feature_extractor(model, return_nodes=[module_name])
    feature_map = extractor(image)[module_name][0, :, :].detach()[kernel_idx]

    plt.figure()
    plt.imshow(feature_map, cmap='gray')
    plt.axis("off")
    
    if save is not None:
        plt.savefig(f'{save}/feature_maps_{module_name}_k{kernel_idx}.png')
    else:
        plt.show()
    
    plt.close()


# Adapted from: https://github.com/justinbellucci/cnn-visualizations-pytorch/tree/master
def visualize_activation_maximization(model, epochs, module_name, kernel_idx, save=None):
    
    grad = {}
        
    def get_grad(name):
        def hook(model, grad_in, grad_out):
            grad[name] = grad_out[0,kernel_idx]
        return hook

    names = module_name.split(sep='.')
    module = reduce(getattr, names, model)
    module.register_forward_hook(get_grad("conv"))    
    extractor = create_feature_extractor(model, return_nodes=[module_name])

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    data_transform = Compose(
        [
            Resize(224,224),
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
        x = extractor(x)
    
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
                plt.savefig(f'{save}/activation_max_{module_name}_k{kernel_idx}_e{epoch}.png')
            else:
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
    parser.add_argument("--save")
    parser.add_argument("--config")
    parser.add_argument("--module_name")
    parser.add_argument("--kernel_idx", type=int)
    parser.add_argument("--n_class", type=int)
    parser.add_argument("--pretrained", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    
    if args.device == "cpu":
        device = torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    data_transform = Compose(
        [
            Resize(224,224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
    )
    
    model = get_model(args.model, args.n_class, args.pretrained)
    
    model.load_state_dict(torch.load(args.config, map_location=device))
    model.eval()
    
    #print(model)
    
    img = np.array(Image.open(args.image))
    
    if args.save is None:
        plt.imshow(img)
        plt.show()
        plt.close()
    
    img = data_transform(image=img)['image']
    img = img.unsqueeze(0)

    # Example calls
    visualize_kernel(model, args.module_name, args.kernel_idx, args.save)
    visualize_feature_map(model, img, args.module_name, args.kernel_idx, args.save)
    visualize_activation_maximization(model, 50, args.module_name, args.kernel_idx, args.save)
    visualize_saliency_map(model, img, args.save)
