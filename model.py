import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A lightweight convolutional neural network (CNN) for image classification.

    The network consists of three convolutional blocks (Conv + ReLU + MaxPool),
    followed by adaptive average pooling and a small fully-connected classifier.

    Adaptive pooling is used so the model does not depend on a fixed input
    resolution (it will work with any reasonably sized square image).
    """

    def __init__(self, n_classes: int) -> None:
        """
        Initialize the CNN architecture.

        Parameters
        ----------
        n_classes : int
            Number of target classes for classification (size of the output layer).

        Returns
        -------
        None
            This constructor initializes the model modules.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Removes the dependency on a fixed spatial size (e.g. 128x128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """
        Run a forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of images with shape (batch_size, 3, H, W).

        Returns
        -------
        torch.Tensor
            Logits tensor with shape (batch_size, n_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
