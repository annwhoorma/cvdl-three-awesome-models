# Three awesome models

Assignment 2 for Computer Vision &amp; Deep Learning course at Innopolis University. README contains a very short version of the report which you can by downloading _A2_report.pdf_

## Shortcut: Links to Colab Notebooks

It's highly advised to view Jupyter Notebooks from Colab because GitHub does not display some outputs.

[YOLOv4-tiny](https://colab.research.google.com/drive/1wYGkd6upzia8fPyI-Ft6TA2_ASqCqhou?usp=sharing) |
[YOLOv5-tiny](https://colab.research.google.com/drive/1t-sz0c1Jal0283zRVZLB_wBa-pdkfeRM?usp=sharing) |
[MaskRCNN](https://colab.research.google.com/drive/10N_UdY9q9dUKKxfRsnpGJpAbRoeMrIoK?usp=sharing)

_parse_dataset.py_ is a script responsible for convertation of supervisely format to the format accepted by MaskRCNN.

## Project Idea

This project is aimed to detect recycling codes: PP (5), PAP (20-22), ALU (41). To achive that, I use three models: YOLOv4, YOLOv5, and MaskRCNN. You can find all Jupyter notebooks and resulting folders at this [Google Drive folder](https://drive.google.com/drive/folders/1gHMC1etvBosvCI_ABcaIyHHHY1AYIrHd?usp=sharing).

## Data Acquisition and Annotation

I took photos of whatever I found at home. You can see examples below:

Since I didn't have much of aluminium stuff, the dataset is a bit unbalanced. The statistics is the following:

| Class name     | Images count | Objects count |
| -------------- | ------------ | ------------- |
| **PAP**        | 31           | 32            |
| **POL**        | 36           | 37            |
| **ALU**        | 29           | 29            |

The numbers in _Objects Count_ column is different from the numbers in _Objects Count_ column because some images contain more than 1 class.

<div style="display: flex; align-items: center; justify-content: center;">
<img src="images/1.jpg" style="width: 19%;"/>
<img src="images/2.jpg" style="width: 19%;"/>
<img src="images/3.jpg" style="width: 19%;"/>
<img src="images/4.jpg" style="width: 19%;"/>
<img src="images/5.jpg" style="width: 19%;"/>
</div>

I used [supervisely](https://app.supervise.ly/) to create polygon-annotations for the images. I present the examples of annotated objects below.

<div style="display: flex; align-items: center; justify-content: center;">
<img src="images/1_a.png" style="width: 19%;"/>
<img src="images/2_a.png" style="width: 19%;"/>
<img src="images/3_a.png" style="width: 19%;"/>
<img src="images/4_a.png" style="width: 19%;"/>
<img src="images/5_a.png" style="width: 19%;"/>
</div>

## Data Preprocessing and Augmentation

### For YOLOv4 and YOLOv5
I used [roboflow](https://roboflow.com/) for data augmentation for YOLOs.

**Preprocessing**
- Auto-Orient: Applied

**Augmentations**
- Outputs per training example: 3
- Rotation: Between -45?? and +45??
- Shear: ??20?? Horizontal, ??20?? Vertical
- Hue: Between -180?? and +180??
- Saturation: Between -50% and +50%
- Brightness: Between -30% and +30%
- Blur: Up to 2.75px

I present the examples of augmentated images below.

<div style="display: flex; align-items: center; justify-content: center;">
<img src="images/1_r.png" style="width: 19%;"/>
<img src="images/2_r.png" style="width: 19%;"/>
<img src="images/3_r.png" style="width: 19%;"/>
<img src="images/4_r.png" style="width: 19%;"/>
<img src="images/5_r.png" style="width: 19%;"/>
</div>

### For MaskRCNN

I used [supervisely](https://app.supervise.ly/) to augment the dataset and increase its size. I added augmentations by using supervisely's DTL language to write a config and run the job. The result of this is a new dataset that has 234 images.

**Augmentations**:
- Resize: 700x700, keep aspect ratio
- Rotate: Between -180?? and +180??
- Gaussian Blur: sigma between 0.5 and 2
- Contrast: between 0.5 and 2
- Brightness: between -50 and 50
- Random Color

<div style="display: flex; align-items: center; justify-content: center;">
<img src="images/1_s.png" style="width: 19%;"/>
<img src="images/2_s.png" style="width: 19%;"/>
<img src="images/3_s.png" style="width: 19%;"/>
<img src="images/4_s.png" style="width: 19%;"/>
<img src="images/5_s.png" style="width: 19%;"/>
</div>

I got the following statistics for this dataset:

| Class name     | Images count | Objects count |
| -------------- | ------------ | ------------- |
| **PAP**        | 93           | 96            |
| **POL**        | 108          | 111           |
| **ALU**        | 87           | 87            |

Finally, to get the required data representation, I've written my own python script _parse_dataset.py_ that takes a path to dataset and creates two _.json_ files: one for training and one for validation - by default _train.json_ and _valid.json_ respectively.

## Training YOLOv4

I used tiny-config and Darknet data format to train YOLOv4. As described before, I used Roboflow to convert dataset in Supervisely format to dataset in Darknet format.

- YOLOv4 trained for 6000 iterations
- Last accuracy: 79.66%; best accuracy: 83.51%
- The model saw 360000 images

| CLass Name | Average Precision, % | True Positive (TP) | False Positive (FP) |
| ---------- | -------------------- | ------------------ | ------------------- |
| **ALU**    | 70.24                | 4                  | 0                   |
| **PAP**    | 67.26                | 5                  | 3                   |
| **POL**    | 100.00               | 8                  | 0                   |

As I wrote before, POL class had the highest number of images and PP recyling code is always associated with 5, whereas PAP and ALU codes could be associated with multiple numbers. Probably that is why AP for POL class is 100%.

### Input

<img src="images/yolov4/yolov4_input.png"/>

### Output

<img src="images/yolov4/yolov4_output.png"/>

## Training YOLOv5 with initial weights

I used tiny-config and pytorch-yolo data format to train YOLOv5. As described before, I used Roboflow to convert dataset in Supervisely format to dataset in yolov5 format.

- YOLOv5 trained for 1750 epochs (early stopping due to small improvement over the last 1000 epochs);
- Last precision: 92%; Last recall: 45%; Last mAP: 48%;
- Reached 1400th epoch after 2hr 16mins - very slow!

### Prediction on test set (best weights)

First and second images do not have any predictions at all. Third and forth have correct predictions.

<img src="images/yolov5/pretrained_prediction.png"/>

### Comparison

I also trained a YOLO without pretrained weights, below you can see a comparison. Both results are frustrating.

| YOLOv5 type           | Precision | Recall | mAP  |
| --------------------- | --------- | ------ | ---- |
| **YOLOv5**            | 86%       | 47%    | 29%  |
| **Pretrained YOLOv5** | 92%       | 45%    | 48%  |


## Training MaskRCNN

These few lines contain a dense and most important information about configuration for the model:
1. I use pretrained weights from `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`.
2. Initial learning rate is 0.0002
3. Number of iterations: 4000.
4. Decrease learning rate by 0.5 at iterations: 2800, 3600.
5. Images size is 700x700.
6. Finished after 58 minutes - very fast!

I have already described the way I augmented and transformed the data for MaskRCNN and visualized it, so just enjoy the statistics and results! :)

## Metrics

Evaluation results for segmentation:
|   AP   |  AP50   |  AP75   |  APs   |  APm   |  APl   |
|--------|---------|---------|--------|--------|--------|
| 89.765 | 100.000 | 100.000 | 85.050 | 90.040 | 94.175 |

Per-category segm AP:
| Class Name     | AP     |
|----------------|--------|
| **PAP**        | 90.644 |
| **POL**        | 88.651 |
| **ALU**        | 90.000 |

It is very interesting that here POL does not have the heighest score, as we saw that it has a little bit more smaples than other classes.

### Output

I am very satisfied with the results! Maybe the augmentation from Supervisely played a role here, but...
1. The training of MaskRCNN was faster
2. The results are awesome
3. The model is confident
4. The masks are ultra fitting

<div style="display: flex; align-items: center; justify-content: center;">
<img src="images/maskrcnn/pred1.png" style="width: 19%;"/>
<img src="images/maskrcnn/pred2.png" style="width: 19%;"/>
<img src="images/maskrcnn/pred3.png" style="width: 19%;"/>
<img src="images/maskrcnn/pred4.png" style="width: 19%;"/>
<img src="images/maskrcnn/pred5.png" style="width: 19%;"/>
</div>
