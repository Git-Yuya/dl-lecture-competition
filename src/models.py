import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128) -> None:
        super().__init__()  # Calling the constructor of the parent class

        # Defining the convolutional blocks and the classification head
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Adaptive average pooling
            Rearrange("b d 1 -> b d"),  # Rearranging dimensions for linear layer
            nn.Linear(hid_dim, num_classes),  # Linear layer for classification
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            X (b, c, t): Input tensor.
        Returns:
            X (b, num_classes): Output tensor after classification.
        """
        X = self.blocks(X)  # Passing input through convolutional blocks

        return self.head(X)  # Returning the output after classification


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()  # Calling the constructor of the parent class
        
        self.in_dim = in_dim  # Setting the input dimension
        self.out_dim = out_dim  # Setting the output dimension

        # Defining the convolutional layers, batch normalization, and dropout
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # Applying skip connection if input and output dimensions are the same
        else:
            X = self.conv0(X)  # Convolutional operation

        X = F.gelu(self.batchnorm0(X))  # Applying activation function and batch normalization

        X = self.conv1(X) + X  # Applying skip connection
        X = F.gelu(self.batchnorm1(X))  # Applying activation function and batch normalization

        X = self.conv2(X) + X  # Applying skip connection
        X = F.gelu(self.batchnorm2(X))  # Applying activation function and batch normalization

        return self.dropout(X)  # Applying dropout and returning the output
