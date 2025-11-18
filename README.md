# VGG16 Retina OCT Classification

Transfer learning pipeline for classifying retinal OCT (Optical Coherence Tomography) images into four disease categories using a VGG16-BN backbone in PyTorch.

> CNV • DME • DRUSEN • NORMAL

This repository contains the training, evaluation and simple inference scripts used for experimenting with the OCT2017 retinal dataset.

The pre-trained model file vgg16_bn-6c64b313.pth is shared by the link https://drive.google.com/file/d/1asCNcKLiEDx2RpWHAijaj-x1VxGs76pB/view?usp=sharing

---

## Project Structure

```text
VGG_16_model_training/
├── OCT_VGG-16.py           # Main training script (train/val/test, save best model)
├── OCT_VGG-16_Test.py      # Evaluation + visualization on the OCT2017 test split
├── TEST.py                 # Simple inference on images inside ./test/
├── OCT2017_/               # Dataset root (train / val / test folders)
│   ├── train/
│   ├── val/
│   └── test/
│       ├── CNV/
│       ├── DME/
│       ├── DRUSEN/
│       └── NORMAL/
├── test/                   # Small folder with sample JPEGs for quick inference demo
│   ├── CNV-....jpeg
│   ├── DME-....jpeg
│   ├── DRUSEN-....jpeg
│   └── NORMAL-....jpeg
└── vgg16bn/
    └── vgg16_bn-6c64b313.pth   # ImageNet-pretrained VGG16-BN weights
```

---

## Requirements

- Python ≥ 3.8
- [PyTorch](https://pytorch.org/)
- torchvision
- numpy
- matplotlib
- Pillow (PIL)

Install the basic dependencies:

```bash
pip install torch torchvision matplotlib pillow numpy
```

A CUDA-capable GPU is strongly recommended, but the scripts will also run on CPU (more slowly).

---

## Dataset Setup (`OCT2017_`)

The code assumes the following directory layout inside `OCT2017_`:

```text
OCT2017_/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

Each subfolder should contain the corresponding class JPEG images.  
Class order (hard-coded in the scripts):

```python
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
```

> **Note:** The dataset itself is **not** included in this repository.  
> Please download the OCT retinal dataset you are using (commonly referred to as *OCT2017*) and place it under `OCT2017_/` with the structure above.

---

## Pretrained VGG16-BN Weights

`OCT_VGG-16.py` and `OCT_VGG-16_Test.py` load ImageNet-pretrained weights from:

```python
vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load("vgg16bn/vgg16_bn-6c64b313.pth"))
```

You can either:

1. Download `vgg16_bn-6c64b313.pth` from the official PyTorch model zoo and place it in the `vgg16bn/` folder, **or**
2. Modify the code to use torchvision’s built-in weights, for example:

```python
vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
```

---

## Training

The main training script is **`OCT_VGG-16.py`**.

It performs:

- Data loading with augmentation (random crop, rotation, color jitter, horizontal flip) for `train`
- Center-crop evaluation pipelines for `val` and `test`
- Transfer learning on VGG16-BN:
  - Replace the final classifier layer with a 4-class output
  - Optimize all parameters with SGD (lr=1e-4, momentum=0.9)
  - StepLR scheduler (step_size=12, gamma=0.1)
- Reporting loss and accuracy for each epoch
- Evaluation on the test split
- Saving the trained model

Run:

```bash
python OCT_VGG-16.py
```

After training, the model weights are saved as:

```text
VGG16_OCT_Retina_trained_model.pt
```

---

## Evaluation & Visualization on OCT2017 Test Split

**`OCT_VGG-16_Test.py`** uses the same DataLoader setup as `OCT_VGG-16.py` but focuses on:

- Evaluating the trained checkpoint on `OCT2017_/test`
- Printing average loss and accuracy
- Visualizing a batch of test images together with:
  - Ground-truth labels
  - Predicted labels

Example usage:

```bash
python OCT_VGG-16_Test.py
```

---

## Simple Inference on Images in `./test/`

**`TEST.py`** is a lightweight script for quickly testing the saved model on JPEG images in the `test/` folder.

The script:

1. Loads the trained checkpoint:

   ```python
   model_path = 'VGG16_OCT_Retina_trained_model.pt'
   ```

2. Applies the same Resize → CenterCrop → ToTensor transforms used during training.
3. Loops over all `.jpeg` files in `./test/`.
4. Displays a single matplotlib panel where each image is shown with its predicted class title.

Run:

```bash
python TEST.py
```

You should see a window like:

- Row of input OCT images
- Title on each subplot: `Predicted Class: CNV / DME / DRUSEN / NORMAL`

---

## Tips

- If you encounter CUDA memory issues, reduce the batch size in the DataLoader definitions.
- To run on CPU only, you can override `use_gpu = False` in the scripts.
- For your own experiments, feel free to:
  - Change the learning rate / epochs
  - Swap optimizers
  - Add more data augmentation
  - Replace VGG16-BN with another backbone.

---

## 简要中文说明

这是一个使用 **VGG16-BN + 迁移学习** 做视网膜 OCT 四分类的小项目：  
`OCT_VGG-16.py` 用 OCT2017 数据集进行训练并保存模型，  
`OCT_VGG-16_Test.py` 在官方 test 集上做评估和可视化，  
`TEST.py` 在 `test/` 文件夹里的若干 JPEG 图像上做快速推理并画出预测结果。

你只需要：

1. 准备好 `OCT2017_` 数据集目录（train/val/test 四个类别的子文件夹）。
2. 下载并放好 `vgg16bn/vgg16_bn-6c64b313.pth` 预训练权重。
3. 运行 `python OCT_VGG-16.py` 训练，生成 `VGG16_OCT_Retina_trained_model.pt`。
4. 用 `python TEST.py` 或 `python OCT_VGG-16_Test.py` 做测试和可视化即可。

---

## License

This repository is provided for research and educational purposes.  
Please make sure you also respect the license/terms of use of the OCT retinal dataset you download.
