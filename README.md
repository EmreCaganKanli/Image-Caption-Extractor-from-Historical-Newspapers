# Image-Caption-Extractor-from-Historical-Newspapers
Pipeline for extracting images and captions from digitized historical newspapers.

This repository contains the code used to fine-tune a LayoutParser Faster R-CNN model on the [FINLAM Dataset](https://huggingface.co/datasets/Teklia/Newspapers-finlam).

---

## Requirements
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [PyTorch](https://pytorch.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with the `pytesseract` wrapper

---

## Training
1. Clone [Detectron2](https://github.com/facebookresearch/detectron2) into a folder named `detectron2`.  
2. Download pretrained weights for LayoutParser trained on the Newspaper Navigator dataset from the [LayoutParser Model Zoo](https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html).  
3. Prepare the dataset:  
   - Use the COCO-style annotations under `coco_annotations/`.  
   - Copy FINLAM dataset images into three folders:  
     - `train_photos/`  
     - `validation_photos/`  
     - `test_photos/`  
4. Run the Jupyter notebook `fine-tune.ipynb` to fine-tune the model.

---

## Inference
- **`fine_tune_eval.py`**  
  - Evaluates the fine-tuned model and outputs AP scores for the chosen dataset.  
  - Also saves prediction visualizations:  
    - Validation set → `output/val_visualizations/`  
    - Test set → `output/test_visualizations/`  
  - In this repo, 5 sample images from each folder are already included for reference.  

- **`data_creator_single.py`**  
  - For a single newspaper page, detects illustrations and selects the **three closest caption regions per illustration**.  
  - Saves crops of the illustration and caption regions into `cropped_images/`.  
  - Writes a `.txt` file with Tesseract OCR outputs for the extracted captions.  

- **`data_creator.py`**  
  - Same as above, but runs on an entire dataset instead of a single newspaper page.  

---

## Results
- The fine-tuned model achieved **46.6 mAP** on the FINLAM test set.  
- Example visualizations of model predictions can be found in `output/val_visualizations/` and `output/test_visualizations/` (5 images each included in this repo).  

---

## Acknowledgements
- [Detectron2](https://github.com/facebookresearch/detectron2)  
- [LayoutParser](https://layout-parser.readthedocs.io/)  
- [FINLAM Dataset](https://huggingface.co/datasets/Teklia/Newspapers-finlam)  
