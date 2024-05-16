import os
import csv
import argparse
import torch
from ultralytics import YOLO
from PIL import Image

IOU = 0.8
MAX_DET = 3
MIN_SIZE = 56

def iterative_detection(device, model, minimal_conf, data_dir):
    
    for filename in os.listdir(data_dir):      
        
        if os.path.isfile(f"{data_dir}/{filename}"):
            
            results = model.predict(f"{data_dir}/{filename}", imgsz=640, device=device, conf=minimal_conf, classes=[2,5,7], iou=IOU, max_det=MAX_DET)
    
            for i, r in enumerate(results):
                print(r.boxes.data.tolist())
                # Plot results image
                im_bgr = r.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    
                # Show results to screen (in supported environments)
                r.show()
    
        lock = input("<<<Press enter to continue or type '0' to finish>>>")
        
        if lock == '0':
            break

def create_dataset(device, model, minimal_conf, scale, in_dir, out_dir):
    
    for filename in os.listdir(in_dir):  
            
        results = model.predict(f"{in_dir}/{filename}", imgsz=640, device=device, conf=minimal_conf, augment=True, classes=[2,5,7], iou=IOU, max_det=MAX_DET)
        img = Image.open(f"{in_dir}/{filename}")

        for i, boxes in enumerate(results[0].boxes):
            for box in boxes:
                coords = tuple(map(int, box.xyxy[0].cpu()))
                width, height = abs(coords[0] - coords[2]), abs(coords[1] - coords[3])
                
                if width >= MIN_SIZE and height >= MIN_SIZE:
                
                    cropped_image = img.crop(coords)
                    cropped_image = cropped_image.resize(scale, Image.BICUBIC)
                    cropped_image.save(f"{out_dir}/{filename}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--indir")
    parser.add_argument("--outdir")
    parser.add_argument("--minimal_conf", type=float)
    parser.add_argument("--scale", type=int)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    scale = (args.scale, args.scale)
    
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    #iterative_detection(device, model, args.minimal_conf, args.indir)
    create_dataset(device, model, args.minimal_conf, scale, args.indir, args.outdir)
