import os
import csv
import argparse

from ultralytics import YOLO
from PIL import Image

#MIN_CONF_DAY = 0.7
#MIN_CONF_NIGHT = 0.6
IOU = 0.8
MAX_DET = 3
MIN_SIZE = 56

def iterative_detection(device, model, minimal_conf, data_dir):
    for filename in os.listdir(root_dir):      
        
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
    
    #if not os.path.exists(out_dir):
    #    os.mkdir(out_dir)

    for filename in os.listdir(in_dir):  
            
        results = model.predict(f"{in_dir}/{filename}", imgsz=640, device=device, conf=minimal_conf, augment=True, classes=[2,5,7], iou=IOU, max_det=MAX_DET)
        img = Image.open(f"{in_dir}/{filename}")

        for i, boxes in enumerate(results[0].boxes):
            for box in boxes:
                coords = tuple(map(int, box.xyxy[0].cpu()))
                width, height = abs(coords[0] - coords[2]), abs(coords[1] - coords[3])
                
                if width >= MIN_SIZE and height >= MIN_SIZE:
                
                    cropped_image = img.crop(coords)
                    cropped_image.resize(scale, Image.BICUBIC)
                    cropped_image.save(f"{out_dir}/{filename}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--in_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--min_conf", type=float)
    parser.add_argument("--scale", type=int)
    args = parser.parse_args()
    
    device = args.device
    in_dir = args.in_dir
    out_dir = args.out_dir
    min_conf = args.min_conf
    scale = (args.scale, args.scale)
    
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    #iterative_detection(device, model, root_dir)
    create_dataset(device, model, min_conf, scale, in_dir, out_dir)
