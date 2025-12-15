import torch
import torch.nn as nn
import logging
from typing import List, Optional

class FusionMLP(nn.Module):
    """Multi-Layer Perceptron for fusing DCT and FFT frequency features.
    Provides more expressive feature fusion than simple linear projection."""

    def __init__(self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """Initialize FusionMLP.
        Parameters:
            input_dim (int): input feature dimension
            output_dim (int): output feature dimension
            hidden_dims (List[int], optional): dimensions of hidden layers. If None, uses [input_dim, input_dim // 2]
            dropout_rate (float): dropout probability for regularization
            use_batch_norm (bool): whether to use batch normalization
        Raises:
            ValueError: if parameters are invalid
        """
        super(FusionMLP, self).__init__()

        self.logger: logging.Logger = logging.getLogger("frequal.FusionMLP")
        """Logger instance for the FusionMLP class"""

        try:
            assert input_dim > 0, f"Input dimension must be positive, got {input_dim}"
            assert output_dim > 0, f"Output dimension must be positive, got {output_dim}"
            assert 0.0 <= dropout_rate < 1.0, f"Dropout rate must be in [0, 1), got {dropout_rate}"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        if hidden_dims is None:
            hidden_dims = [input_dim, input_dim // 2]

        self.input_dim: int = input_dim
        """Input feature dimension"""
        self.output_dim: int = output_dim
        """Output feature dimension"""
        self.hidden_dims: List[int] = hidden_dims
        """Hidden layer dimensions"""

        #build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        #output layer (no activation, batch norm, or dropout)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp: nn.Sequential = nn.Sequential(*layers)
        """Sequential MLP layers"""

        self.logger.info(f"FusionMLP initialized: {input_dim} -> {hidden_dims} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP.
        Parameters:
            x (torch.Tensor): input features [batch, input_dim]
        Returns:
            torch.Tensor: fused features [batch, output_dim]
        """
        return self.mlp(x)