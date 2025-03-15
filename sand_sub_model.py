
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

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
DATASET_PATH = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Sand"
IMAGES_PATH = DATASET_PATH
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "Sand_PASCAL_VOC")
OUTPUT_LABELS_PATH = os.path.join(DATASET_PATH, "labels")
os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)

# Extract all unique class names from XML files
def extract_class_names(xml_folder):
    class_names = set()
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
CLASS_NAMES = extract_class_names(ANNOTATIONS_PATH)
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
xml_files = glob(os.path.join(ANNOTATIONS_PATH, "*.xml"))
for xml_file in xml_files:
    convert_voc_to_yolo(xml_file, OUTPUT_LABELS_PATH, CLASS_NAMES)

print("Conversion to YOLO format completed!")

# Organize train/val split
YOLO_IMAGES = DATASET_PATH
YOLO_LABELS = OUTPUT_LABELS_PATH
TRAIN_IMAGES = os.path.join(DATASET_PATH, "images/train")
VAL_IMAGES = os.path.join(DATASET_PATH, "images/val")
TRAIN_LABELS = os.path.join(DATASET_PATH, "labels/train")
VAL_LABELS = os.path.join(DATASET_PATH, "labels/val")
os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

image_files = glob(os.path.join(YOLO_IMAGES, "*.jpg"))
if len(image_files) == 0:
    raise ValueError("No image files found. Check the directory path and file extensions.")
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

def move_files(file_list, source_folder, target_folder):
    for file_path in file_list:
        filename = os.path.basename(file_path)
        label_file = os.path.splitext(filename)[0] + ".txt"
        shutil.copy(file_path, target_folder)
        shutil.copy(os.path.join(source_folder, label_file), target_folder.replace("images", "labels"))

move_files(train_images, YOLO_LABELS, TRAIN_IMAGES)
move_files(val_images, YOLO_LABELS, VAL_IMAGES)

print("Dataset organized into train/val split!")

# Create data.yaml
data_yaml = f"""
train: {TRAIN_IMAGES}
val: {VAL_IMAGES}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

with open(os.path.join(DATASET_PATH, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("data.yaml file created!")

# Train YOLOv8
model = YOLO('yolov8n.pt')
model.train(data=os.path.join(DATASET_PATH, "data.yaml"), epochs=30, imgsz=416, batch=16, name="yolov8_sand", workers=2)
print("Training completed!")

# Save Trained Weights to Google Drive
MODEL_DIR = "/content/runs/detect/yolov8_sand/weights/"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last.pt")

# Define Google Drive save location
GDRIVE_SAVE_PATH = "/content/drive/MyDrive/yolov8_sand"
os.makedirs(GDRIVE_SAVE_PATH, exist_ok=True)

# Copy best and last model weights to Google Drive
shutil.copy(BEST_MODEL_PATH, os.path.join(GDRIVE_SAVE_PATH, "yolov8_sand_best.pt"))
shutil.copy(LAST_MODEL_PATH, os.path.join(GDRIVE_SAVE_PATH, "yolov8_sand_last.pt"))

print("Trained weights saved to Google Drive successfully!")

# Visualization Functions
def visualize_image_with_labels(image_path, label_path, class_names, title="Ground Truth"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(label_path, "r") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.split())
            class_id = int(class_id)
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            label = class_names[class_id]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

def visualize_image_with_predictions(image_path, results, class_names, title="Model Predictions"):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    for bbox, score, class_id in zip(detections, scores, class_ids):
        x_min, y_min, x_max, y_max = map(int, bbox)
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Visualize 10 Random Validation Images
random.seed(42)
selected_images = random.sample(val_images, 10)

print("\nVisualizing 10 Validation Images with Ground Truth...")
for image_path in selected_images:
    label_path = os.path.join(VAL_LABELS, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    visualize_image_with_labels(image_path, label_path, CLASS_NAMES, title="Validation - Ground Truth")

print("\nVisualizing 10 Validation Images with Model Predictions...")
for image_path in selected_images:
    results = model.predict(source=image_path, save=False)
    visualize_image_with_predictions(image_path, results, CLASS_NAMES, title="Validation - Model Predictions")


# Export the model
model.export(format='onnx')
print("Model exported!")


##------extra Visualization-----#

for i, image_path in enumerate(val_images[:10]):  # Limit to 10 images
    print(f"Running inference on Image {i + 1}/{len(val_images)}: {image_path}")
    results = model.predict(source=image_path, conf=0.25, save=False)  # Adjust confidence threshold

    # Iterate over results (typically one result per image)
    for result in results:
        # Extract predictions
        if result.boxes:
            detections = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        else:
            detections, scores, class_ids = [], [], []

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(detections) > 0:
            for bbox, score, class_id in zip(detections, scores, class_ids):
                x_min, y_min, x_max, y_max = map(int, bbox)
                color = (0, 255, 0)  # Green for predictions
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add a label if no detections
        else:
            cv2.putText(image, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the image with predictions
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(f"Validation - Model Predictions (Image {i + 1})")
        plt.axis("off")
        plt.show()
