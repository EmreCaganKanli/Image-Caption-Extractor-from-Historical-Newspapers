if __name__ == "__main__":
    import os
    import torch
    import cv2
    import numpy as np
    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data import build_detection_test_loader, MetadataCatalog
    from detectron2.modeling.postprocessing import detector_postprocess
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"./Tesseract-OCR/tesseract.exe"

    
    register_coco_instances("newspaper_val", {}, "coco_annotations/val.json", "photos")
    MetadataCatalog.get("newspaper_val").thing_classes = ["Illustration", "Title"]

    # Set up config
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"
    cfg.DATASETS.TEST = ("newspaper_val",)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Build model (no TTA)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().to(cfg.MODEL.DEVICE)

    # Load metadata
    metadata = MetadataCatalog.get("newspaper_val")
    class_names = metadata.thing_classes

    # Inference on dataset
    val_loader = build_detection_test_loader(cfg, "newspaper_val")

    os.makedirs("cropped_boxes", exist_ok=True)

    image_counter = 0
    for inputs in val_loader:
        if image_counter >= 20:
            break

        with torch.no_grad():
            outputs = model(inputs)

        for i in range(len(inputs)):
            if image_counter >= 20:
                break

            image_path = inputs[i]["file_name"]
            image = cv2.imread(image_path)

            if image is None:
                "No image found for " + inputs[i]["file_name"]
                continue

            instances = detector_postprocess(
                outputs[i]["instances"],
                inputs[i]["height"],
                inputs[i]["width"]
            ).to("cpu")

            scores = instances.scores.numpy()
            boxes = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes.numpy()
            title_boxes = []
            illustration_boxes = []

            for j in range(len(boxes)):
                if scores[j] < 0.5:
                    continue 
                
                class_name = class_names[classes[j]]
                box = boxes[j].tolist()
                if class_name == "Title":
                    title_boxes.append(box)
                elif class_name == "Illustration":
                    illustration_boxes.append(box)
                score = scores[j]

            for ill_i, ill in enumerate(illustration_boxes):
                x1, y1, x2, y2 = int(ill[0]), int(ill[1]), int(ill[2]), int(ill[3])
                cropped = image[y1:y2, x1:x2]
                save_path = os.path.join("cropped_boxes", inputs[i]["file_name"][7:-4] + f"_illustration{ill_i}.jpg")
                cv2.imwrite(save_path, cropped)
                distance_dict = {}
                center = [ill[2] - ill[0], ill[3] - ill[1]]

                for title_i, title in enumerate(title_boxes):
                    center2 = [title[2] - title[0], title[3] - title[1]]
                    distance_dict[title_i] = np.sqrt(((center[0] - center2[0])**2)*10 + (center[1] - center2[1])**2)
                closest_titles_ids = sorted(distance_dict, key=distance_dict.get)[:3]
                closest_titles = [title_boxes[indx] for indx in closest_titles_ids]

                
                for title_i, title in enumerate(closest_titles):
                    x1, y1, x2, y2 = int(title[0]), int(title[1]), int(title[2]), int(title[3])
                    cropped = image[y1:y2, x1:x2]
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    text = pytesseract.image_to_string(gray, lang='fra')
                    text_path = os.path.join("cropped_boxes", inputs[i]["file_name"][7:-4] + ".txt")
                    with open(text_path, "a") as f:
                        f.write(f"{text.strip()}\n")
                    save_path = os.path.join("cropped_boxes", inputs[i]["file_name"][7:-4] + f"_illustration{ill_i}_title{title_i}.jpg")
                    cv2.imwrite(save_path, cropped)
            image_counter += 1
