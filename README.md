# Image Classification with a Convolutional Neural Network (PyTorch)

This project implements an end-to-end image classification pipeline using a
convolutional neural network (CNN) in PyTorch. It covers dataset loading,
image preprocessing, model training, evaluation, and result visualization.

The codebase is structured to separate data handling, model definition, and
training logic, following machine learning engineering practices.

---

## Project structure

- `train.py` – Main training and evaluation script  
- `data.py` – Dataset loading and image transformations  
- `model.py` – CNN architecture  
- `utils.py` – Plotting and helper functions  
- `README.md` – Project documentation  

---

## Dataset format

The dataset must be organized into three splits: training, validation, and test.

```
data/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
```

Each `<class_name>` directory represents one class.  
All image files (e.g. `.jpg`, `.png`, `.jpeg`) inside that directory belong to that class.  
The same class names must appear in `train`, `val`, and `test`.

---

## Installation

Install the required Python packages:

```bash
pip install torch torchvision matplotlib scikit-learn
```

---

## How to run

Run the training and evaluation pipeline with:

```bash
python train.py \
  --data-dir data \
  --img-size 128 \
  --batch-size 32 \
  --epochs 10 \
  --outdir OUT
```

---

## Command-line arguments

| Argument | Description |
|--------|-------------|
| `--data-dir` | Path to the dataset root directory containing `train`, `val`, and `test`. |
| `--img-size` | All images are resized to this size (e.g. 128×128) before being passed to the network. |
| `--batch-size` | Number of images processed in parallel during training and evaluation. |
| `--epochs` | Number of training epochs (full passes over the training set). |
| `--outdir` | Directory where all outputs (models, metrics, and plots) are saved. |

---

## Output files

After training finishes, the output directory will contain:

- `cnn_model.pt` – Trained PyTorch model weights  
- `classification_report.txt` – Precision, recall, and F1-score per class  
- `metrics.json` – Accuracy and confusion matrix in machine-readable form  
- `figures/01_train_class_counts.png` – Class distribution of the training set  
- `figures/02_confusion_matrix.png` – Confusion matrix on the test set  



---


