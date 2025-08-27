import os
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.modeling.postprocessing import detector_postprocess
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"./Tesseract-OCR/tesseract.exe"

# Put your image path here
image_path = "./1.png"

# Load model config
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333

# Load model
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
model.eval().to(cfg.MODEL.DEVICE)

# Set class names manually
MetadataCatalog.get("custom_inference").thing_classes = ["Illustration", "Title"]
class_names = MetadataCatalog.get("custom_inference").thing_classes

# Read image
image = cv2.imread(image_path)

# Prepare image
height, width = image.shape[:2]
input_tensor = torch.as_tensor(image.transpose(2, 0, 1)).float().to(cfg.MODEL.DEVICE)
inputs = [{"image": input_tensor, "height": height, "width": width}]

# Run inference
with torch.no_grad():
    outputs = model(inputs)

instances = detector_postprocess(outputs[0]["instances"], height, width).to("cpu")
scores = instances.scores.numpy()
boxes = instances.pred_boxes.tensor.numpy()
classes = instances.pred_classes.numpy()

title_boxes = []
illustration_boxes = []

# Separate boxes
for j in range(len(boxes)):
    if scores[j] < 0.5:
        continue
    class_name = class_names[classes[j]]
    box = boxes[j].tolist()
    if class_name == "Title":
        title_boxes.append(box)
    elif class_name == "Illustration":
        if scores[j] < 0.8:
            continue
        illustration_boxes.append(box)

# Prepare output folder
os.makedirs("cropped_boxes", exist_ok=True)
base_filename = os.path.splitext(os.path.basename(image_path))[0]

def calculate_center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

def is_intersecting(boxA, boxB, ratio=0.4):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    heightA = y2A - y1A
    heightB = y2B - y1B

    # Epsilon is relative to smaller height
    eps = ratio * min(heightA, heightB)

    # Relaxed separation check
    horizontal_separate = (x2A < x1B - eps) or (x2B < x1A - eps)
    vertical_separate   = (y2A < y1B - eps) or (y2B < y1A - eps)

    return not (horizontal_separate or vertical_separate)

# Process each illustration
for ill_i, ill in enumerate(illustration_boxes):
    x1, y1, x2, y2 = map(int, ill)
    cropped = image[y1:y2, x1:x2]
    save_path = os.path.join("cropped_boxes", f"{aaa}.jpg")
    aaa += 1
    cv2.imwrite(save_path, cropped)

    center = calculate_center(ill)
    distance_dict = {}

    for title_i, title in enumerate(title_boxes):
        center2 = calculate_center(title)
        dist = np.sqrt(((center[0] - center2[0])**2)*10 + (center[1] - center2[1])**2)

        ((center[0] - center2[0])**2)*10 + (center[1] - center2[1])**2

        distance_dict[title_i] = dist

    closest_titles_ids_sorted = sorted(distance_dict, key=distance_dict.get)
    closest_titles_ids_temp = [[], [], []]
    for title_id in closest_titles_ids_sorted:
        brk = False
        for title_set in closest_titles_ids_temp:
            if title_set == []:
                title_set.append(title_id)
                break
            for title_in_set in title_set:
                if is_intersecting(title_boxes[title_in_set], title_boxes[title_id]):
                    title_set.append(title_id)
                    brk = True
                    break
            if brk:
                break

    print(closest_titles_ids_temp)
    closest_titles = []
    for title_set in closest_titles_ids_temp:
        combined_box = [min(title_boxes[b][i] for b in title_set) if i < 2 else max(title_boxes[b][i] for b in title_set) for i in range(4)]
        closest_titles.append(combined_box)

    for title_i, title in enumerate(closest_titles):
        x1, y1, x2, y2 = map(int, title)
        cropped_title = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_title, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang="fra")

        # Save cropped title image
        title_img_path = os.path.join("cropped_boxes", f"{aaa}.jpg")
        aaa += 1
        cv2.imwrite(title_img_path, cropped_title)

        # Save OCR text
        text_file = os.path.join("cropped_boxes", f"{base_filename}.txt")
        with open(text_file, "a", encoding="utf-8") as f:
            f.write(f"{text.strip()}\n")
            f.write("------------------------\n")

