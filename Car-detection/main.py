import os
import csv
import argparse

from ultralytics import YOLO
from PIL import Image

MIN_CONF_DAY = 0.65
MIN_CONF_NIGHT = 0.4

def iterative_detection(root_dir):
    for filename in os.listdir(root_dir):      
        
        if os.path.isfile(f"{root_dir}/{filename}"):
            
            results = model.predict(f"{root_dir}/{filename}", imgsz=640, device=1, classes=[2,5,7])
    
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

def create_dataset(device, model, reference_file, root_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(reference_file) as file:
        csv_reader = csv.reader(file, delimiter=',')

        # row   := [image_filename,label]
        # label := 0 -> day; 1 -> night; 2 -> dawn/dusk; 3 -> undefined 
        for row in csv_reader:
            
            minimal_conf = MIN_CONF_DAY
            
            if row[1] == 1 or row[1] == 2:
                minimal_conf = MIN_CONF_NIGHT    
            
            elif row[1] == 3:
                continue
            
            results = model.predict(f"{root_dir}/{row[0]}", imgsz=640, device=device, augment=True, classes=[2,5,7])
            
            img = Image.open(f"{root_dir}/{row[0]}")

            for i, boxes in enumerate(results[0].boxes):
                for box in boxes:
                    coords = tuple(map(int, box.xyxy[0].cpu()))
                    cropped_image = img.crop(coords)
                    cropped_image.save(f"{output_dir}/{row[0].split('.')[0]}-{i}_{row[1]}.jpg")
                    #cropped_image.show()                    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--reference_file")
    parser.add_argument("--root_dir")
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    
    device = args.device
    reference_file = args.reference_file
    root_dir = args.root_dir
    output_dir = args.output_dir
    
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    create_dataset(device, model, reference_file, root_dir, output_dir)