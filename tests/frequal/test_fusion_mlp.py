"""Tests for FusionMLP."""
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from frequal import FusionMLP


class TestFusionMLPInitialization:
    """Test FusionMLP initialization."""

    def test_init_with_default_hidden_dims(self):
        """Test initialization with default hidden dimensions."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        assert mlp.input_dim == 1024
        assert mlp.output_dim == 512
        assert mlp.hidden_dims == [1024, 512]  #default is [input_dim, input_dim // 2]

    def test_init_with_custom_hidden_dims(self):
        """Test initialization with custom hidden dimensions."""
        hidden_dims = [512, 256, 128]
        mlp = FusionMLP(input_dim=1024, output_dim=64, hidden_dims=hidden_dims)
        assert mlp.hidden_dims == hidden_dims

    def test_init_with_batch_norm(self):
        """Test initialization with batch normalization."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, use_batch_norm=True)
        #check that LayerNorm layers are present (uses LayerNorm instead of BatchNorm for batch_size=1 support)
        has_layer_norm = any(isinstance(layer, nn.LayerNorm) for layer in mlp.mlp)
        assert has_layer_norm

    def test_init_without_batch_norm(self):
        """Test initialization without batch normalization."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, use_batch_norm=False)
        #check that no normalization layers are present
        has_batch_norm = any(isinstance(layer, nn.BatchNorm1d) for layer in mlp.mlp)
        has_layer_norm = any(isinstance(layer, nn.LayerNorm) for layer in mlp.mlp)
        assert not has_batch_norm
        assert not has_layer_norm

    def test_init_with_dropout(self):
        """Test initialization with dropout."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, dropout_rate=0.5)
        #check that Dropout layers are present
        dropout_layers = [layer for layer in mlp.mlp if isinstance(layer, nn.Dropout)]
        assert len(dropout_layers) > 0
        assert dropout_layers[0].p == 0.5

    def test_init_invalid_input_dim(self):
        """Test initialization with invalid input dimension."""
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            FusionMLP(input_dim=0, output_dim=512)

    def test_init_invalid_output_dim(self):
        """Test initialization with invalid output dimension."""
        with pytest.raises(ValueError, match="Output dimension must be positive"):
            FusionMLP(input_dim=1024, output_dim=-1)

    def test_init_invalid_dropout_rate(self):
        """Test initialization with invalid dropout rate."""
        with pytest.raises(ValueError, match="Dropout rate must be in"):
            FusionMLP(input_dim=1024, output_dim=512, dropout_rate=1.5)


class TestFusionMLPForward:
    """Test FusionMLP forward pass."""

    def test_forward_output_shape(self):
        """Test that forward pass produces correct output shape."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        x = torch.randn(8, 1024)
        output = mlp(x)
        assert output.shape == (8, 512)

    def test_forward_single_sample(self):
        """Test forward pass with single sample."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        mlp.eval()  #use eval mode to avoid BatchNorm issue with single sample
        x = torch.randn(1, 1024)
        output = mlp(x)
        assert output.shape == (1, 512)

    def test_forward_large_batch(self):
        """Test forward pass with large batch."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        x = torch.randn(128, 1024)
        output = mlp(x)
        assert output.shape == (128, 512)

    def test_forward_with_different_hidden_dims(self):
        """Test forward pass with different hidden layer configurations."""
        hidden_dims_list = [
            [512, 256],
            [800, 600, 400],
            [2048]
        ]
        for hidden_dims in hidden_dims_list:
            mlp = FusionMLP(input_dim=1024, output_dim=128, hidden_dims=hidden_dims)
            x = torch.randn(8, 1024)
            output = mlp(x)
            assert output.shape == (8, 128)

    def test_forward_deterministic_in_eval_mode(self):
        """Test that forward pass is deterministic in eval mode."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, dropout_rate=0.5)
        mlp.eval()
        x = torch.randn(8, 1024)
        output1 = mlp(x)
        output2 = mlp(x)
        assert torch.allclose(output1, output2)

    def test_forward_different_in_train_mode_with_dropout(self):
        """Test that forward pass is different in train mode with dropout."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, dropout_rate=0.5)
        mlp.train()
        x = torch.randn(8, 1024)
        output1 = mlp(x)
        output2 = mlp(x)
        #outputs should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the MLP."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        x = torch.randn(8, 1024, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_forward_output_range(self):
        """Test that output values are in reasonable range."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        x = torch.randn(8, 1024)
        output = mlp(x)
        #output should not contain NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestFusionMLPArchitecture:
    """Test FusionMLP architecture properties."""

    def test_layer_count(self):
        """Test that the correct number of layers is created."""
        hidden_dims = [512, 256]
        mlp = FusionMLP(input_dim=1024, output_dim=128, hidden_dims=hidden_dims, use_batch_norm=True)
        linear_layers = [layer for layer in mlp.mlp if isinstance(layer, nn.Linear)]
        #should have len(hidden_dims) + 1 linear layers (hidden + output)
        assert len(linear_layers) == len(hidden_dims) + 1

    def test_activation_functions(self):
        """Test that ReLU activation functions are present."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        relu_layers = [layer for layer in mlp.mlp if isinstance(layer, nn.ReLU)]
        #should have ReLU after each hidden layer (not after output)
        assert len(relu_layers) >= 1

    def test_no_activation_after_output_layer(self):
        """Test that there's no activation after the output layer."""
        mlp = FusionMLP(input_dim=1024, output_dim=512, hidden_dims=[512])
        #last layer should be Linear
        assert isinstance(mlp.mlp[-1], nn.Linear)

    def test_parameter_count(self):
        """Test that MLP has learnable parameters."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        param_count = sum(p.numel() for p in mlp.parameters())
        assert param_count > 0

    def test_parameters_are_trainable(self):
        """Test that all parameters require gradients."""
        mlp = FusionMLP(input_dim=1024, output_dim=512)
        for param in mlp.parameters():
            assert param.requires_grad
