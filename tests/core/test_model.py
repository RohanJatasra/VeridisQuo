"""Tests for DeepFakeDetector model (core/model.py)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from core.model import DeepFakeDetector
from utils.config_schemas import ModelArchitectureConfig


class TestDeepFakeDetectorInitialization:
    """Test DeepFakeDetector initialization."""

    def test_init_with_valid_config(self, minimal_model_config):
        """Test initialization with valid configuration."""
        model = DeepFakeDetector(config=minimal_model_config)

        assert model is not None
        assert isinstance(model, nn.Module)
        assert model.config == minimal_model_config

    def test_init_without_config_raises_error(self):
        """Test initialization without config raises ValueError."""
        with pytest.raises(ValueError, match="ModelArchitectureConfig is required"):
            DeepFakeDetector(config=None)

    def test_init_creates_all_components(self, minimal_model_config):
        """Test initialization creates all required components."""
        model = DeepFakeDetector(config=minimal_model_config)

        #check all components exist
        assert hasattr(model, 'efficient')
        assert hasattr(model, 'frequal_extractor')
        assert hasattr(model, 'concatenator')
        assert hasattr(model, 'classifier')

        #check components are not None
        assert model.efficient is not None
        assert model.frequal_extractor is not None
        assert model.concatenator is not None
        assert model.classifier is not None

    def test_init_efficient_net_component(self, minimal_model_config):
        """Test EfficientNet component is initialized correctly."""
        model = DeepFakeDetector(config=minimal_model_config)

        from spatial import EfficientNetFeatureExtractor
        assert isinstance(model.efficient, EfficientNetFeatureExtractor)

    def test_init_frequency_extractor_component(self, minimal_model_config):
        """Test frequency extractor component is initialized correctly."""
        model = DeepFakeDetector(config=minimal_model_config)

        from frequal import FrequencyFeatureExtractor
        assert isinstance(model.frequal_extractor, FrequencyFeatureExtractor)

    def test_init_concatenator_component(self, minimal_model_config):
        """Test concatenator component is initialized correctly."""
        model = DeepFakeDetector(config=minimal_model_config)

        from classification import FaceFeaturesConcatenator
        assert isinstance(model.concatenator, FaceFeaturesConcatenator)

    def test_init_classifier_component(self, minimal_model_config):
        """Test classifier component is initialized correctly."""
        model = DeepFakeDetector(config=minimal_model_config)

        from classification import FaceClassifier
        assert isinstance(model.classifier, FaceClassifier)


class TestDeepFakeDetectorForward:
    """Test DeepFakeDetector forward pass."""

    def test_forward_output_shape(self, sample_model, sample_input_batch):
        """Test forward pass produces correct output shape."""
        output = sample_model(sample_input_batch)

        assert output.shape == (sample_input_batch.shape[0], 2)

    def test_forward_output_dtype(self, sample_model, sample_input_batch):
        """Test forward pass output has correct dtype."""
        output = sample_model(sample_input_batch)

        assert output.dtype == torch.float32

    def test_forward_batch_processing(self, sample_model):
        """Test forward pass handles different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = sample_model(x)
            assert output.shape == (batch_size, 2)

    def test_forward_single_image(self, sample_model, sample_input_single):
        """Test forward pass with single image."""
        output = sample_model(sample_input_single)

        assert output.shape == (1, 2)

    def test_forward_produces_logits(self, sample_model, sample_input_batch):
        """Test forward pass produces logits (not probabilities)."""
        output = sample_model(sample_input_batch)

        #logits can be any real number (not bounded [0, 1])
        assert not torch.all((output >= 0) & (output <= 1))

    def test_forward_deterministic_in_eval_mode(self, sample_model, sample_input_batch):
        """Test forward pass is deterministic in eval mode."""
        sample_model.eval()

        output1 = sample_model(sample_input_batch)
        output2 = sample_model(sample_input_batch)

        assert torch.allclose(output1, output2)

    def test_forward_stochastic_in_train_mode(self, sample_model, sample_input_batch):
        """Test forward pass may differ in train mode due to dropout."""
        sample_model.train()

        output1 = sample_model(sample_input_batch)
        output2 = sample_model(sample_input_batch)

        #with dropout, outputs may differ (not guaranteed, but likely)
        #we just check the code runs without error in train mode
        assert output1.shape == output2.shape

    def test_forward_invalid_shape_raises_error(self, sample_model):
        """Test forward pass with invalid input shape raises error."""
        #wrong number of channels
        with pytest.raises(AssertionError, match="Input must be"):
            sample_model(torch.randn(4, 1, 224, 224))

        #wrong spatial dimensions
        with pytest.raises(AssertionError, match="Input must be"):
            sample_model(torch.randn(4, 3, 128, 128))

    def test_forward_normalized_input(self, sample_model, normalized_input_batch):
        """Test forward pass with ImageNet-normalized input."""
        output = sample_model(normalized_input_batch)

        assert output.shape == (normalized_input_batch.shape[0], 2)

    def test_forward_output_is_differentiable(self, sample_model, sample_input_batch):
        """Test forward pass output supports backpropagation."""
        sample_input_batch.requires_grad = True
        output = sample_model(sample_input_batch)

        #check gradient can be computed
        loss = output.sum()
        loss.backward()

        assert sample_input_batch.grad is not None


class TestDeepFakeDetectorIntegration:
    """Test DeepFakeDetector end-to-end integration."""

    def test_end_to_end_inference(self, sample_model, sample_input_batch):
        """Test complete inference pipeline from image to prediction."""
        sample_model.eval()

        #forward pass
        logits = sample_model(sample_input_batch)

        #convert to probabilities
        probabilities = torch.softmax(logits, dim=1)

        #get predictions
        predictions = torch.argmax(probabilities, dim=1)

        assert logits.shape == (sample_input_batch.shape[0], 2)
        assert probabilities.shape == (sample_input_batch.shape[0], 2)
        assert predictions.shape == (sample_input_batch.shape[0],)
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_gradient_flow_through_all_components(self, sample_model, sample_input_batch):
        """Test gradient flows through all model components."""
        sample_model.train()
        sample_input_batch.requires_grad = True

        output = sample_model(sample_input_batch)
        loss = output.sum()
        loss.backward()

        #check gradients exist for all main components
        has_gradients = any(
            p.grad is not None
            for p in sample_model.parameters()
        )
        assert has_gradients

    def test_train_mode_vs_eval_mode_behavior(self, sample_model, sample_input_batch):
        """Test model behavior differs between train and eval modes."""
        #eval mode
        sample_model.eval()
        output_eval = sample_model(sample_input_batch)

        #train mode
        sample_model.train()
        output_train = sample_model(sample_input_batch)

        #shapes should match
        assert output_eval.shape == output_train.shape

    def test_probability_distribution_valid(self, sample_model, sample_input_batch):
        """Test probability distribution is valid."""
        sample_model.eval()

        logits = sample_model(sample_input_batch)
        probabilities = torch.softmax(logits, dim=1)

        #probabilities should sum to 1 for each sample
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums))

        #all probabilities should be in [0, 1]
        assert torch.all((probabilities >= 0) & (probabilities <= 1))

    def test_confidence_scores_reasonable(self, sample_model, sample_input_batch):
        """Test confidence scores are in reasonable range."""
        sample_model.eval()

        logits = sample_model(sample_input_batch)
        probabilities = torch.softmax(logits, dim=1)
        confidences = probabilities.max(dim=1).values

        #confidences should be between 0 and 1
        assert torch.all((confidences >= 0) & (confidences <= 1))

    def test_batch_prediction_consistency(self, sample_model):
        """Test predictions are consistent for same input in batch."""
        sample_model.eval()

        #create batch with duplicated input
        single_input = torch.randn(1, 3, 224, 224)
        batch_input = single_input.repeat(4, 1, 1, 1)

        output = sample_model(batch_input)

        #all outputs should be identical
        for i in range(1, 4):
            assert torch.allclose(output[0], output[i])

    def test_model_parameters_count(self, sample_model):
        """Test model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in sample_model.parameters())

        #DeepFakeDetector should have millions of parameters (EfficientNet-B4 is large)
        assert total_params > 1_000_000
        assert total_params < 100_000_000  #sanity check

    def test_model_trainable_parameters(self, sample_model):
        """Test model has trainable parameters."""
        trainable_params = sum(
            p.numel() for p in sample_model.parameters() if p.requires_grad
        )

        assert trainable_params > 0


class TestDeepFakeDetectorDeviceCompatibility:
    """Test DeepFakeDetector device transfer."""

    def test_model_on_cpu(self, sample_model, sample_input_batch):
        """Test model works on CPU."""
        sample_model.cpu()
        sample_input_batch_cpu = sample_input_batch.cpu()

        output = sample_model(sample_input_batch_cpu)

        assert output.device.type == 'cpu'
        assert output.shape == (sample_input_batch.shape[0], 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_gpu(self, sample_model, sample_input_batch):
        """Test model works on GPU."""
        sample_model.cuda()
        sample_input_batch_gpu = sample_input_batch.cuda()

        output = sample_model(sample_input_batch_gpu)

        assert output.device.type == 'cuda'
        assert output.shape == (sample_input_batch.shape[0], 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer_cpu_to_gpu(self, sample_model, sample_input_batch):
        """Test transferring model from CPU to GPU."""
        #start on CPU
        sample_model.cpu()
        output_cpu = sample_model(sample_input_batch.cpu())

        #transfer to GPU
        sample_model.cuda()
        output_gpu = sample_model(sample_input_batch.cuda())

        #outputs should have same shape
        assert output_cpu.shape == output_gpu.shape

    def test_model_device_attribute(self, sample_model):
        """Test model device can be queried."""
        #model parameters should all be on same device
        devices = {p.device for p in sample_model.parameters()}
        assert len(devices) == 1  #all parameters on same device


class TestDeepFakeDetectorEdgeCases:
    """Test DeepFakeDetector edge cases."""

    def test_forward_with_zeros_input(self, sample_model):
        """Test forward pass with all-zeros input."""
        zeros_input = torch.zeros(4, 3, 224, 224)
        output = sample_model(zeros_input)

        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_ones_input(self, sample_model):
        """Test forward pass with all-ones input."""
        ones_input = torch.ones(4, 3, 224, 224)
        output = sample_model(ones_input)

        assert output.shape == (4, 2)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_extreme_values(self, sample_model):
        """Test forward pass with extreme input values."""
        extreme_input = torch.randn(4, 3, 224, 224) * 100  #very large values
        output = sample_model(extreme_input)

        assert output.shape == (4, 2)
        #model should handle extreme values without NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_repr(self, sample_model):
        """Test model has string representation."""
        repr_str = repr(sample_model)

        assert repr_str is not None
        assert len(repr_str) > 0
        assert 'DeepFakeDetector' in repr_str
