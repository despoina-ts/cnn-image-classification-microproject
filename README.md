# CNN Image Classification Microproject (PyTorch)

This repository contains a complete deep learning pipeline for image
classification using a Convolutional Neural Network (CNN) implemented
in PyTorch.

The project performs dataset exploration, CNN training and model
evaluation on an image classification task.

---

## Repository contents

- `cnn_microproject_torch.py`  
  Main script that loads the dataset, trains a CNN and evaluates it.

---

## Dataset format

data/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg

 
Each class folder should contain image files (e.g. `.jpg`, `.png`, `.jpeg`).

---

## Requirements

- Python 3  
- torch  
- torchvision  
- numpy  
- matplotlib  
- scikit-learn  

Install all dependencies with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn

```
--- 
How to run 

python cnn_microproject_torch.py --data-dir data --img-size 128 --epochs 10 --outdir OUT
