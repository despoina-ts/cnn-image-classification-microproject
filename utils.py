from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Path to the directory that should be created.

    Returns
    -------
    None
        This function does not return anything. It creates the directory
        as a side effect if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_class_counts(dataset, out_png: str, title: str) -> None:
    """
    Plot the number of samples per class in a dataset.

    This function counts how many images belong to each class and
    saves a bar plot to disk. It is useful for visualizing class
    imbalance in the training data.

    Parameters
    ----------
    dataset : torchvision.datasets.ImageFolder
        Dataset object containing image samples and class labels.
    out_png : str
        Path where the generated plot will be saved.
    title : str
        Title of the plot.

    Returns
    -------
    None
        The plot is saved to disk. No value is returned.
    """
    counts = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1

    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Images")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_confusion(cm, class_names, out_png: str) -> None:
    """
    Save a confusion matrix plot to disk.

    Parameters
    ----------
    cm : numpy.ndarray
        Confusion matrix computed from model predictions.
    class_names : list of str
        Names of the classes corresponding to the matrix axes.
    out_png : str
        Path where the generated plot will be saved.

    Returns
    -------
    None
        The confusion matrix figure is saved to disk.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
