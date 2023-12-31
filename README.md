# YOLOv8 Object Detection with Augmented Dataset

This project focuses on training a YOLOv8 object detection model using various image augmentation techniques and leveraging the prepared dataset. The trained model is then used for testing on both videos and images for object detection tasks.

## Project Overview

- **Dataset Preparation:**
  - Utilized multiple image augmentation libraries such as Imgaug, Albumentations, SOLT, and Tensor to augment and enhance the dataset.
  - Employed the LabelImg library for precise dataset labeling, ensuring accurate annotations for object detection.

- **Model Training:**
  - Trained a YOLOv8 model using the augmented and labeled dataset.
  - Utilized transfer learning techniques to fine-tune the pre-trained YOLOv8 architecture on the customized dataset.

- **Testing and Evaluation:**
  - Performed testing on various videos and images using the trained YOLOv8 model.
  - Evaluated the model's performance in object detection tasks across different media.

## Project Structure

- `dataset/`: Contains augmented and labeled dataset.
- `mymodel.ipynb.url`: Here is the colab link of Training and Testing Code.
- `inference/`: Results and findings from testing on videos and images.
- `source/`: inputs

## Usage

1. **Dataset Preparation:**
   - Augment the dataset using any preferred augmentation library.
   - Label the augmented images precisely using LabelImg.

2. **Model Training:**
   - Train the YOLOv8 model using the augmented and labeled dataset.
   - Save the trained model weights in the `model/` directory(google drive).

3. **Testing:**
   - Utilize the trained model for object detection on videos and images.
   - Record observations and results in the `testing/` directory(google drive).

## Requirements

- Python 3.x
- YOLOv8 framework
- Image augmentation libraries (Imgaug, Albumentations, SOLT, Tensor)
- LabelImg library for dataset labeling

## Resources

-Link is Mensioned above

## Contributor

- Tosif Ansari
