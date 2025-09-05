import argparse
import os
import torch
import yaml
from tqdm import tqdm

# --- Import from our libs ---
from lib.dataset import InferenceDataset, multitask_collate_fn
from lib.device import select_device
from lib.model import build_model
from lib.checkpoint import CheckpointManager

# Note: classify_transforms is not in your libs, assuming it's available
from ultralytics.data.augment import classify_transforms

def load_config(path):
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-task inference script using the modular library.")
    # source and weights are now in the config file
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the model and data configuration file.")
    return parser.parse_args()

def main():
    """Main function to perform inference on a directory of images."""
    args = parse_args()
    cfg = load_config(args.config)
    
    # Deconstruct config for clarity
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    inference_cfg = cfg['inference']

    device = select_device(inference_cfg.get('device', 'cuda'))
    task_classes = data_cfg['task_classes']
    task_names = sorted(list(task_classes.keys())) # Sort for consistent output order

    #  Dataset and dataLoader
    source_dir = inference_cfg['source_dir']
    transform = classify_transforms()
    inference_ds = InferenceDataset(source_image_dir=source_dir, transform=transform)
    
    inference_loader = torch.utils.data.DataLoader(
        inference_ds,
        batch_size=inference_cfg['batch_size'],
        shuffle=False,
        collate_fn=multitask_collate_fn,
        num_workers=inference_cfg.get('num_workers', 4),
        pin_memory=True
    )

    # Model Building
    print(f"Building model: {model_cfg['name']}...")
    model = build_model(model_name=model_cfg['name'], task_classes=task_classes, freeze=False, device=device)

    weights_path = inference_cfg['weights_path']
    manager = CheckpointManager() 
    
    manager.load(path=weights_path, device=device, model=model)
    # legacy code -> previous checkpoints only saved model weights (default torch.save call) 
    #CheckpointManager._load_legacy(weights_path, model, device)
    
    model.eval()
    
    print(f"Model weights loaded from '{weights_path}'. Model is in evaluation mode.")

    # Inference Loop
    with torch.no_grad():
        for imgs, _, fnames in tqdm(inference_loader, desc="Running Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            
            # Process results for each image in the batch
            for i in range(len(imgs)):
                pred_tuple, conf_tuple = [], []
                filename = fnames[i]

                for task in task_names:
                    logits = outputs[task][i]
                    probs = torch.softmax(logits, dim=0)
                    pred = probs.argmax().item()
                    conf = probs[pred].item()

                    pred_tuple.append(pred)
                    conf_tuple.append(round(conf, 4))
                
                # Print results in a CSV-friendly format
                print(f"{filename},{str(tuple(pred_tuple)).replace(' ', '')},{str(tuple(conf_tuple)).replace(' ', '')}")

if __name__ == '__main__':
    main()