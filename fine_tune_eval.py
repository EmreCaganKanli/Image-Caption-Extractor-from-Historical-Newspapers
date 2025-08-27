if __name__ == "__main__":
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.datasets import register_coco_instances
    import os
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    import torch
    from detectron2.modeling.postprocessing import detector_postprocess
    from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA

    CONFIDENCE_THRESHOLD = 0
    mode = "test"
    dataset_mode = "newspaper_" + mode

    register_coco_instances("newspaper_train", {}, "coco_annotations/train.json", "train_photos")
    register_coco_instances("newspaper_val", {}, "coco_annotations/val.json", "val_photos")
    register_coco_instances("newspaper_test", {}, "coco_annotations/test.json", "test_photos")

    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda"

    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.MIN_SIZES = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    cfg.TEST.AUG.MAX_SIZE = 4000

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.DATASETS.TRAIN = ("newspaper_train",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = "./output"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD 

    MetadataCatalog.get("newspaper_train").thing_classes = ["Illustration", "Title"]
    MetadataCatalog.get(dataset_mode).thing_classes = ["Illustration", "Title"]

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.DATASETS.TEST = (dataset_mode,)
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model = GeneralizedRCNNWithTTA(cfg, model)


    evaluator = COCOEvaluator(dataset_mode, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, dataset_mode)

    results = inference_on_dataset(model, val_loader, evaluator)
    print(results)

    # Create output dir
    os.makedirs("output/" + mode + "_visualizations", exist_ok=True)

    metadata = MetadataCatalog.get(dataset_mode)
    model.eval()
    model = model.to(cfg.MODEL.DEVICE)

    image_counter = 0  # ✅ Add this before the loop
            
    for idx, inputs in enumerate(val_loader):
        with torch.no_grad():
            outputs = model(inputs)

        for i in range(len(inputs)):
            image_path = inputs[i]["file_name"]
            original_img = cv2.imread(image_path)

            processed_output = detector_postprocess(
                outputs[i]["instances"], 
                inputs[i]["height"], 
                inputs[i]["width"]
            )

            # ✅ Filter by confidence threshold
            instances = processed_output.to("cpu")
            keep = instances.scores > 0.5
            filtered_instances = instances[keep]

            # Visualize
            v = Visualizer(original_img[:, :, ::-1], metadata=metadata, scale=1.0)
            v = v.draw_instance_predictions(filtered_instances)

            # Save image
            filename = os.path.basename(image_path)
            out_path = os.path.join("output/" + mode + "_visualizations", filename)
            cv2.imwrite(out_path, v.get_image()[:, :, ::-1])

            image_counter += 1                           
            if image_counter >= 20: break                
        if image_counter >= 20: break                    

