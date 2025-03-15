
import os
import random
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import yaml
import glob
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Helper Functions
# -------------------------------

def load_gt_boxes(label_path, image_size):
    """
    Loads ground-truth boxes from a YOLO-format label file.
    Each line: class x_center y_center width height (normalized).
    Returns a numpy array of boxes: [x1, y1, x2, y2].
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, xc, yc, w, h = map(float, parts)
                    W, H = image_size
                    x1 = (xc - w/2) * W
                    y1 = (yc - h/2) * H
                    x2 = (xc + w/2) * W
                    y2 = (yc + h/2) * H
                    boxes.append([x1, y1, x2, y2])
    return np.array(boxes)

def create_dataset_yaml(condition, images_folder, nc=10, names=["class0", "class1", "class2", "class3", "class4",
                                                              "class5", "class6", "class7", "class8", "class9"]):
    """
    Creates a temporary YAML file for evaluation.
    Sets both "train" and "val" keys to the validation folder and includes "nc" and "names".
    """
    parent_dir = os.path.dirname(os.path.dirname(images_folder))
    data = {
         "path": parent_dir,
         "train": os.path.join("images", "val"),
         "val": os.path.join("images", "val"),
         "nc": nc,
         "names": names
    }
    yaml_path = f"temp_{condition}.yaml"
    with open(yaml_path, "w") as f:
         yaml.dump(data, f)
    return yaml_path

def evaluate_model_on_dataset(model, yaml_path):
    """
    Uses YOLOv8's built-in validation method to evaluate the model on the dataset defined in yaml_path.
    Returns a dictionary with mAP (using map50), precision, recall, and F1 score.
    Note: We access metric values without calling them as functions.
    """
    metrics = model.val(data=yaml_path, verbose=False)
    try:
        mAP = metrics.box.map50   # Access mAP at IoU=0.5
        prec = metrics.box.mp     # Mean precision
        rec = metrics.box.mr      # Mean recall
    except Exception as e:
        print("Error evaluating metrics:", e)
        mAP, prec, rec = 0, 0, 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"mAP": mAP, "precision": prec, "recall": rec, "f1": f1}

def measure_inference_time(model, image_paths):
    """
    Measures average inference time per image for the given model.
    """
    times = []
    for img_path in image_paths:
        start = time.time()
        _ = model(img_path, verbose=False)
        times.append(time.time() - start)
    return np.mean(times)

def measure_classification_time(classifier, image_paths):
    """
    Measures average classification time per image for the visibility classifier.
    """
    times = []
    for img_path in image_paths:
        start = time.time()
        _ = classify_visibility(img_path, classifier)
        times.append(time.time() - start)
    return np.mean(times)

# -------------------------------
# 2. Visibility Classifier & Model Loading
# -------------------------------

class VisibilityClassifier(nn.Module):
    def __init__(self, num_classes=5):  # classes: fog, rain, sand, snow, good_condition_best
        super(VisibilityClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def load_visibility_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisibilityClassifier().to(device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = { k if k.startswith("model.") else f"model.{k}": v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def classify_visibility(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    condition_map = ["fog", "rain", "sand", "snow", "good_condition_best"]
    return condition_map[pred]

# -------------------------------
# 3. YOLO Model Loading
# -------------------------------

BASE_WEIGHTS_PATH = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/weights"
FILE_MAPPING = {
    "fog": ("yolov8_fog", "yolov8_fog_best.pt"),
    "rain": ("yolov8_rain", "yolov8_rain_best.pt"),
    "sand": ("yolov8_sand", "yolov8_sand_best.pt"),
    "snow": ("yolov8_snow", "yolov8_snow_best.pt"),
    "good_condition_best": ("yolov8_good_condition", "yolov8_good_condition_best.pt")
}

def load_submodel(condition):
    if condition not in FILE_MAPPING:
        raise ValueError(f"No submodel mapping for condition={condition}")
    folder, filename = FILE_MAPPING[condition]
    model_path = os.path.join(BASE_WEIGHTS_PATH, folder, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Submodel not found at {model_path}")
    return YOLO(model_path)

# Load the general model (trained on all conditions)
GENERAL_MODEL_PATH = os.path.join(BASE_WEIGHTS_PATH, "yolov8_genral_case", "yolov8_genral_case_best.pt")
general_model = YOLO(GENERAL_MODEL_PATH)

# -------------------------------
# 4. YOLO Inference Function
# -------------------------------

def run_inference(image_path, model):
    results = model(image_path, verbose=False)
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    else:
        boxes, conf, cls_ids = [], [], []
    return {"boxes": boxes, "scores": conf, "class_ids": cls_ids}

# -------------------------------
# 5. Visualization Functions
# -------------------------------

def visualize_comparison_grid(image_paths, sub_detections, gen_detections, sub_model_names, gen_model_names, condition, title_suffix=""):
    """
    Creates a grid with 2 rows and N columns:
      - Top row: images with submodel detections.
      - Bottom row: same images with general model detections.
    """
    N = len(image_paths)
    plt.figure(figsize=(4 * N, 8))
    for i, img_path in enumerate(image_paths):
        orig = cv2.imread(img_path)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img_sub = orig_rgb.copy()
        img_gen = orig_rgb.copy()

        # Draw submodel detections (green)
        for bbox, score, cls in zip(sub_detections[i]["boxes"], sub_detections[i]["scores"], sub_detections[i]["class_ids"]):
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{sub_model_names[cls]}: {score:.2f}"
            cv2.rectangle(img_sub, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_sub, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw general model detections (red)
        for bbox, score, cls in zip(gen_detections[i]["boxes"], gen_detections[i]["scores"], gen_detections[i]["class_ids"]):
            x1, y1, x2, y2 = map(int, bbox)
            label = f"{gen_model_names[cls]}: {score:.2f}"
            cv2.rectangle(img_gen, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_gen, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        plt.subplot(2, N, i+1)
        plt.imshow(img_sub)
        plt.title(f"Submodel ({condition}) {title_suffix}")
        plt.axis("off")

        plt.subplot(2, N, N + i+1)
        plt.imshow(img_gen)
        plt.title(f"General Model {title_suffix}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def select_and_visualize_top_difference(condition, images_folder, num_examples=4, conf_threshold=0.25):
    """
    For a given condition, computes the difference in the count of detections (with confidence >= conf_threshold)
    between the submodel and the general model for each image.
    Selects and displays the top 'num_examples' images where the submodel detects significantly more objects.
    """
    sub_model = load_submodel(condition)
    sub_names = sub_model.names
    gen_names = general_model.names

    image_list = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]

    differences = []
    for img in image_list:
        det_sub = run_inference(img, sub_model)
        det_gen = run_inference(img, general_model)
        count_sub = np.sum(det_sub["scores"] >= conf_threshold)
        count_gen = np.sum(det_gen["scores"] >= conf_threshold)
        diff = count_sub - count_gen
        differences.append((img, diff, det_sub, det_gen))

    # Filter to only images where sub-model detects more objects
    differences = [d for d in differences if d[1] > 0]
    differences.sort(key=lambda x: x[1], reverse=True)

    best = differences[:num_examples]
    if not best:
        print(f"No images found where {condition} submodel detected more objects than the general model.")
        return
    best_imgs = [x[0] for x in best]
    sub_dets = [x[2] for x in best]
    gen_dets = [x[3] for x in best]

    visualize_comparison_grid(best_imgs, sub_dets, gen_dets, sub_names, gen_names, condition, title_suffix="(Top Difference)")

def select_and_visualize_best(condition, images_folder, labels_folder, num_examples=4):
    """
    For a given condition, selects a sample of images (here, the first 'num_examples' from a random sample)
    for side-by-side visualization of submodel vs. general model predictions.
    """
    sub_model = load_submodel(condition)
    sub_names = sub_model.names
    gen_names = general_model.names

    image_list = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]
    sampled_images = random.sample(image_list, min(50, len(image_list)))
    best_comparisons = sampled_images[:num_examples]

    sub_dets = [run_inference(img, sub_model) for img in best_comparisons]
    gen_dets = [run_inference(img, general_model) for img in best_comparisons]

    visualize_comparison_grid(best_comparisons, sub_dets, gen_dets, sub_names, gen_names, condition, title_suffix="(Best Examples)")

# -------------------------------
# 6. Main Evaluation Pipeline: Create Summary Table & Visualize
# -------------------------------

# Define validation folders for each condition (images and labels)
CONDITIONS = {
    "fog": {
        "images": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Fog/images/val",
        "labels": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Fog/labels/val"
    },
    "rain": {
        "images": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Rain/images/val",
        "labels": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Rain/labels/val"
    },
    "sand": {
        "images": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Sand/images/val",
        "labels": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Sand/labels/val"
    },
    "snow": {
        "images": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Snow/images/val",
        "labels": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/Snow/labels/val"
    },
    "good_condition_best": {
        "images": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/good_condition/images/val",
        "labels": "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/good_condition/labels/val"
    }
}

NUM_SAMPLE_IMAGES = 350  # number of images to sample for evaluation
results_records = []

# Load the visibility classifier
visibility_model_path = "/content/drive/MyDrive/DEEP_LEARNING/FinalProject/datasets/DAWN/weights/class_model.pth"
visibility_model = load_visibility_model(visibility_model_path)

# For each condition, evaluate the models and store results
for cond, paths in CONDITIONS.items():
    print(f"\n=== Evaluating condition: {cond.upper()} ===")
    images_folder = paths["images"]
    labels_folder = paths["labels"]

    # Create a temporary YAML file for dataset evaluation with required keys
    yaml_path = create_dataset_yaml(cond, images_folder, nc=10, names=["class0", "class1", "class2", "class3", "class4",
                                                                      "class5", "class6", "class7", "class8", "class9"])

    # Evaluate the condition-specific submodel and general model using YOLO's validation method
    correct_submodel = load_submodel(cond)
    correct_metrics = evaluate_model_on_dataset(correct_submodel, yaml_path)
    general_metrics = evaluate_model_on_dataset(general_model, yaml_path)

    # For the full pipeline, we assume it ideally uses the correct submodel
    full_pipeline_metrics = evaluate_model_on_dataset(load_submodel(cond), yaml_path)

    # Measure inference times on a sample of images
    image_list = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]
    sample_images = random.sample(image_list, min(NUM_SAMPLE_IMAGES, len(image_list)))
    correct_inference_time = measure_inference_time(correct_submodel, sample_images)
    general_inference_time = measure_inference_time(general_model, sample_images)
    classification_time = measure_classification_time(visibility_model, sample_images)
    full_pipeline_inference_time = classification_time + correct_inference_time

    # Compute classification accuracy for the visibility classifier over the sample
    correct_classifications = 0
    for img_path in sample_images:
        predicted_cond = classify_visibility(img_path, visibility_model)
        if predicted_cond == cond:
            correct_classifications += 1
    classification_accuracy = correct_classifications / len(sample_images) if sample_images else 0.0

    results_records.append({
         "Condition": cond,
         "Full Pipeline (mAP)": full_pipeline_metrics["mAP"],
         "Full Pipeline (Classification Accuracy)": classification_accuracy,
         "Full Pipeline (Inference Time, sec)": full_pipeline_inference_time,
         "Correct Sub-model (mAP)": correct_metrics["mAP"],
         "Correct Sub-model (Precision)": correct_metrics["precision"],
         "Correct Sub-model (Recall)": correct_metrics["recall"],
         "Correct Sub-model (F1)": correct_metrics["f1"],
         "Correct Sub-model (Inference Time, sec)": correct_inference_time,
         "General Model (mAP)": general_metrics["mAP"],
         "General Model (Precision)": general_metrics["precision"],
         "General Model (Recall)": general_metrics["recall"],
         "General Model (F1)": general_metrics["f1"],
         "General Model (Inference Time, sec)": general_inference_time
    })

df_results = pd.DataFrame(results_records)
print("\n=== Summary Table ===")
print(df_results)
df_results.to_csv("comparison_results.csv", index=False)

# -------------------------------
# Visualization:
# 1. Visualize images where the submodel detects more objects than the general model.
# -------------------------------
for cond, paths in CONDITIONS.items():
    print(f"\n--- Visualizing images where submodel detects more objects for condition: {cond.upper()} ---")
    select_and_visualize_top_difference(cond, paths["images"], num_examples=6, conf_threshold=0.25)

# -------------------------------
# 2. Visualize best examples for qualitative comparison.
# -------------------------------
for cond, paths in CONDITIONS.items():
    print(f"\n--- Visualizing best examples for condition: {cond.upper()} ---")
    select_and_visualize_best(cond, paths["images"], paths["labels"], num_examples=6)

