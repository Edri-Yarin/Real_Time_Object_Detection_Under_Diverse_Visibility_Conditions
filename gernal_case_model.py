# Import required libraries
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import xmltodict
import random


# Paths
BASE_PATH = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN"
CONDITIONS = ["Rain", "Fog", "Sand", "Snow", "good_condition"]

IMAGES_PATHS = [os.path.join(BASE_PATH, condition) for condition in CONDITIONS]
ANNOTATIONS_PATHS = [os.path.join(path, f"{condition}_PASCAL_VOC") for path, condition in zip(IMAGES_PATHS, CONDITIONS)]
OUTPUT_LABELS_PATH = os.path.join(BASE_PATH, "labels")
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)

# Extract all unique class names from XML files
def extract_class_names(xml_folders):
    class_names = set()
    for xml_folder in xml_folders:
        xml_files = glob(os.path.join(xml_folder, "*.xml"))
        for xml_file in xml_files:
            with open(xml_file) as f:
                xml_content = xmltodict.parse(f.read())
                objects = xml_content["annotation"].get("object", [])
                if not isinstance(objects, list):
                    objects = [objects]
                for obj in objects:
                    class_names.add(obj["name"])
    return sorted(list(class_names))

# Generate CLASS_NAMES dynamically
CLASS_NAMES = extract_class_names(ANNOTATIONS_PATHS)
print(f"Extracted Class Names: {CLASS_NAMES}")

# Convert Pascal VOC XML to YOLO format
def convert_voc_to_yolo(xml_file, output_dir, class_names):
    with open(xml_file) as f:
        xml_content = xmltodict.parse(f.read())
    try:
        image_width = int(xml_content["annotation"]["size"]["width"])
        image_height = int(xml_content["annotation"]["size"]["height"])
    except KeyError as e:
        print(f"Error reading image dimensions in {xml_file}: {e}")
        return

    objects = xml_content["annotation"].get("object", [])
    if not isinstance(objects, list):
        objects = [objects]

    yolo_annotations = []
    for obj in objects:
        class_name = obj["name"]
        if class_name not in class_names:
            print(f"Skipping unrecognized class: {class_name} in {xml_file}")
            continue
        try:
            class_id = class_names.index(class_name)
            bbox = obj["bndbox"]
            x_min, y_min, x_max, y_max = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            bbox_width = (x_max - x_min) / image_width
            bbox_height = (y_max - y_min) / image_height
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
        except KeyError as e:
            print(f"Error reading bounding box in {xml_file}: {e}")
            continue

    if yolo_annotations:
        image_name = os.path.splitext(os.path.basename(xml_file))[0]
        yolo_file_path = os.path.join(output_dir, f"{image_name}.txt")
        with open(yolo_file_path, "w") as f:
            f.write("\n".join(yolo_annotations))

# Convert all XML files to YOLO format
for annotation_path in ANNOTATIONS_PATHS:
    xml_files = glob(os.path.join(annotation_path, "*.xml"))
    for xml_file in xml_files:
        convert_voc_to_yolo(xml_file, OUTPUT_LABELS_PATH, CLASS_NAMES)

print("Conversion to YOLO format completed!")

# Organize train/val split
TRAIN_IMAGES = os.path.join(BASE_PATH, "images/train")
VAL_IMAGES = os.path.join(BASE_PATH, "images/val")
TRAIN_LABELS = os.path.join(BASE_PATH, "labels/train")
VAL_LABELS = os.path.join(BASE_PATH, "labels/val")
os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

image_files = []
for img_path in IMAGES_PATHS:
    image_files.extend(glob(os.path.join(img_path, "*.jpg")))

if len(image_files) == 0:
    raise ValueError("No image files found. Check the directory path and file extensions.")

train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

def move_files(file_list, source_folder, target_folder):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        label_file = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(source_folder, label_file)
        if os.path.exists(label_path):
            shutil.copy(file_path, target_folder)
            shutil.copy(label_path, target_folder.replace("images", "labels"))
        else:
            print(f"Warning: Label file not found for {file_path}")

move_files(train_images, OUTPUT_LABELS_PATH, TRAIN_IMAGES)
move_files(val_images, OUTPUT_LABELS_PATH, VAL_IMAGES)

print("Dataset organized into train/val split!")

# Create data.yaml
data_yaml = f"""
train: {TRAIN_IMAGES}
val: {VAL_IMAGES}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

with open(os.path.join(BASE_PATH, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("data.yaml file created!")

# Train YOLOv8
model = YOLO('yolov8n.pt')
model.train(data=os.path.join(BASE_PATH, "data.yaml"), epochs=30, imgsz=416, batch=16, name="yolov8_all_conditions", workers=2)
print("Training completed!")


import os
import cv2
import random
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define paths
GDRIVE_SAVE_PATH = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/weights/yolov8_genral_case"
BEST_MODEL_PATH = os.path.join(GDRIVE_SAVE_PATH, "yolov8_genral_case_best")

# Ensure model file exists
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}")

# Load the trained model
model = YOLO(BEST_MODEL_PATH)
print("Model loaded successfully!")

# Define validation images path
VAL_IMAGES_PATH = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Rain/images/val"

# Get list of validation images
image_files = [os.path.join(VAL_IMAGES_PATH, f) for f in os.listdir(VAL_IMAGES_PATH) if f.endswith(".jpg")]
if len(image_files) == 0:
    raise ValueError("No validation images found. Check the directory path and file extensions.")

# Select 10 random images
random.seed(42)
selected_images = random.sample(image_files, min(10, len(image_files)))

# Define function to visualize predictions
def visualize_predictions(image_path, results, title="Model Predictions"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    scores = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes is not None else []

    if len(detections) == 0:
        print(f"No detections for image: {image_path}")

    for bbox, score, class_id in zip(detections, scores, class_ids):
        x_min, y_min, x_max, y_max = map(int, bbox)
        label = f"Class {class_id}: {score:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Run validation on selected images
print("Validating on 10 random images...")
for image_path in selected_images:
    print(f"Running inference on: {image_path}")

    # Run inference with lower confidence threshold
    results = model.predict(source=image_path, save=False, conf=0.05)

    # Print raw model output for debugging
    print(f"Detections: {results[0].boxes if results[0].boxes is not None else 'No detections'}")

    # Visualize results
    visualize_predictions(image_path, results, title="Validation - Model Predictions")

print("Validation completed!")

