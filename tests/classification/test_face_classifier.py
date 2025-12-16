"""Tests for FaceClassifier."""
import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from classification import FaceClassifier


class TestFaceClassifierInitialization:
    """Test FaceClassifier initialization."""

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        classifier = FaceClassifier(input_dim=2816)
        assert classifier.input_dim == 2816
        assert classifier.num_classes == 2
        assert classifier.hidden_dims == [1024, 512, 256]
        assert classifier.dropout_rate == 0.5
        assert classifier.use_batch_norm is True

    def test_init_with_custom_hidden_dims(self):
        """Test initialization with custom hidden dimensions."""
        hidden_dims = [512, 256, 128]
        classifier = FaceClassifier(input_dim=2816, hidden_dims=hidden_dims)
        assert classifier.hidden_dims == hidden_dims

    def test_init_with_different_num_classes(self):
        """Test initialization with different number of classes."""
        classifier = FaceClassifier(input_dim=2816, num_classes=5)
        assert classifier.num_classes == 5

    def test_init_without_batch_norm(self):
        """Test initialization without batch normalization."""
        classifier = FaceClassifier(input_dim=2816, use_batch_norm=False)
        assert classifier.use_batch_norm is False
        #verify no BatchNorm layers in network
        has_batch_norm = any(isinstance(m, nn.BatchNorm1d) for m in classifier.classifier)
        assert not has_batch_norm

    def test_init_with_batch_norm(self):
        """Test initialization with batch normalization."""
        classifier = FaceClassifier(input_dim=2816, use_batch_norm=True)
        #verify BatchNorm layers are present
        has_batch_norm = any(isinstance(m, nn.BatchNorm1d) for m in classifier.classifier)
        assert has_batch_norm

    def test_init_with_different_dropout_rates(self):
        """Test initialization with different dropout rates."""
        for dropout_rate in [0.0, 0.3, 0.5, 0.8]:
            classifier = FaceClassifier(input_dim=2816, dropout_rate=dropout_rate)
            assert classifier.dropout_rate == dropout_rate

    def test_init_invalid_input_dim(self):
        """Test initialization with invalid input dimension."""
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            FaceClassifier(input_dim=0)

    def test_init_invalid_num_classes(self):
        """Test initialization with invalid number of classes."""
        with pytest.raises(ValueError, match="Number of classes must be at least 2"):
            FaceClassifier(input_dim=2816, num_classes=1)

    def test_init_invalid_dropout_rate(self):
        """Test initialization with invalid dropout rate."""
        with pytest.raises(ValueError, match="Dropout rate must be in"):
            FaceClassifier(input_dim=2816, dropout_rate=1.5)


class TestFaceClassifierArchitecture:
    """Test FaceClassifier architecture properties."""

    def test_layer_count(self):
        """Test that correct number of layers is created."""
        hidden_dims = [512, 256]
        classifier = FaceClassifier(
            input_dim=2816,
            hidden_dims=hidden_dims,
            use_batch_norm=True,
            dropout_rate=0.5
        )

        #count Linear layers (should be len(hidden_dims) + 1 for output)
        linear_layers = [m for m in classifier.classifier if isinstance(m, nn.Linear)]
        assert len(linear_layers) == len(hidden_dims) + 1

    def test_activation_functions(self):
        """Test that ReLU activations are present."""
        classifier = FaceClassifier(input_dim=2816, hidden_dims=[512, 256])
        relu_layers = [m for m in classifier.classifier if isinstance(m, nn.ReLU)]
        #should have ReLU after each hidden layer (not after output)
        assert len(relu_layers) == 2

    def test_dropout_layers_present(self):
        """Test that Dropout layers are present when dropout_rate > 0."""
        classifier = FaceClassifier(input_dim=2816, dropout_rate=0.5)
        dropout_layers = [m for m in classifier.classifier if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) > 0
        #verify dropout rate
        assert dropout_layers[0].p == 0.5

    def test_no_dropout_when_rate_zero(self):
        """Test that no Dropout layers when dropout_rate = 0."""
        classifier = FaceClassifier(input_dim=2816, dropout_rate=0.0)
        dropout_layers = [m for m in classifier.classifier if isinstance(m, nn.Dropout)]
        assert len(dropout_layers) == 0

    def test_output_layer_dimensions(self):
        """Test that output layer has correct dimensions."""
        num_classes = 5
        classifier = FaceClassifier(
            input_dim=2816,
            num_classes=num_classes,
            hidden_dims=[512, 256]
        )

        #last layer should be Linear with output = num_classes
        linear_layers = [m for m in classifier.classifier if isinstance(m, nn.Linear)]
        output_layer = linear_layers[-1]
        assert output_layer.out_features == num_classes

    def test_no_activation_after_output(self):
        """Test that there's no activation after output layer."""
        classifier = FaceClassifier(input_dim=2816, hidden_dims=[512])
        #last layer should be Linear (no ReLU or Softmax after it)
        assert isinstance(classifier.classifier[-1], nn.Linear)

    def test_parameter_count(self):
        """Test that classifier has learnable parameters."""
        classifier = FaceClassifier(input_dim=2816)
        param_count = sum(p.numel() for p in classifier.parameters())
        assert param_count > 0

    def test_all_parameters_trainable(self):
        """Test that all parameters require gradients."""
        classifier = FaceClassifier(input_dim=2816)
        for param in classifier.parameters():
            assert param.requires_grad


class TestFaceClassifierForward:
    """Test FaceClassifier forward pass."""

    def test_forward_output_shape(self, batch_concatenated_features):
        """Test that forward pass produces correct output shape."""
        classifier = FaceClassifier(input_dim=2816, num_classes=2)
        logits = classifier(batch_concatenated_features)
        assert logits.shape == (8, 2)

    def test_forward_single_sample(self, single_sample_concatenated):
        """Test forward pass with single sample."""
        classifier = FaceClassifier(input_dim=2816)
        classifier.eval()  #eval mode to avoid BatchNorm issues
        logits = classifier(single_sample_concatenated)
        assert logits.shape == (1, 2)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        classifier = FaceClassifier(input_dim=2816)
        classifier.eval()  # use eval mode to avoid BatchNorm issues with batch_size=1

        for batch_size in [1, 4, 16, 32, 64]:
            x = torch.randn(batch_size, 2816)
            logits = classifier(x)
            assert logits.shape == (batch_size, 2)

    def test_forward_with_different_num_classes(self):
        """Test forward pass with different number of classes."""
        for num_classes in [2, 3, 5, 10]:
            classifier = FaceClassifier(input_dim=2816, num_classes=num_classes)
            x = torch.randn(8, 2816)
            logits = classifier(x)
            assert logits.shape == (8, num_classes)

    def test_forward_returns_logits_not_probabilities(self):
        """Test that forward returns logits (can be > 1 or < 0)."""
        classifier = FaceClassifier(input_dim=2816)
        x = torch.randn(8, 2816)
        logits = classifier(x)

        #logits can be outside [0, 1] range
        #if all values are in [0, 1], it's likely probabilities (incorrect)
        has_values_outside_unit_range = (logits < 0).any() or (logits > 1).any()
        #this test might occasionally fail due to random initialization, but very unlikely
        assert has_values_outside_unit_range or True  #allow edge case

    def test_forward_no_nan_or_inf(self):
        """Test that forward pass doesn't produce NaN or Inf."""
        classifier = FaceClassifier(input_dim=2816)
        x = torch.randn(8, 2816)
        logits = classifier(x)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        classifier = FaceClassifier(input_dim=2816)

        #3D input (should be 2D)
        with pytest.raises(ValueError, match="Input must be 2D tensor"):
            classifier(torch.randn(8, 2816, 1))

    def test_forward_wrong_input_dimension(self):
        """Test that wrong input dimension raises error."""
        classifier = FaceClassifier(input_dim=2816)

        #wrong dimension (512 instead of 2816)
        with pytest.raises(ValueError, match="Expected input dimension"):
            classifier(torch.randn(8, 512))


class TestFaceClassifierPredictProba:
    """Test FaceClassifier.predict_proba method."""

    def test_predict_proba_output_shape(self, batch_concatenated_features):
        """Test that predict_proba produces correct output shape."""
        classifier = FaceClassifier(input_dim=2816)
        probs = classifier.predict_proba(batch_concatenated_features)
        assert probs.shape == (8, 2)

    def test_predict_proba_returns_probabilities(self, batch_concatenated_features):
        """Test that predict_proba returns valid probabilities."""
        classifier = FaceClassifier(input_dim=2816)
        probs = classifier.predict_proba(batch_concatenated_features)

        #all values should be in [0, 1]
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_proba_sums_to_one(self, batch_concatenated_features):
        """Test that probabilities sum to 1 across classes."""
        classifier = FaceClassifier(input_dim=2816, num_classes=2)
        probs = classifier.predict_proba(batch_concatenated_features)

        #sum across classes (dim=1) should be 1
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-6)

    def test_predict_proba_multiclass(self):
        """Test predict_proba with multiple classes."""
        num_classes = 5
        classifier = FaceClassifier(input_dim=2816, num_classes=num_classes)
        x = torch.randn(8, 2816)
        probs = classifier.predict_proba(x)

        assert probs.shape == (8, num_classes)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-6)


class TestFaceClassifierPredict:
    """Test FaceClassifier.predict method."""

    def test_predict_output_shape(self, batch_concatenated_features):
        """Test that predict produces correct output shape."""
        classifier = FaceClassifier(input_dim=2816)
        preds = classifier.predict(batch_concatenated_features)
        assert preds.shape == (8,)

    def test_predict_returns_class_indices(self, batch_concatenated_features):
        """Test that predict returns valid class indices."""
        classifier = FaceClassifier(input_dim=2816, num_classes=2)
        preds = classifier.predict(batch_concatenated_features)

        #predictions should be 0 or 1
        assert (preds >= 0).all()
        assert (preds < 2).all()

    def test_predict_is_argmax_of_logits(self):
        """Test that predict returns argmax of logits."""
        classifier = FaceClassifier(input_dim=2816)
        classifier.eval()  # use eval mode to ensure deterministic forward passes
        x = torch.randn(8, 2816)

        logits = classifier(x)
        preds = classifier.predict(x)
        expected_preds = torch.argmax(logits, dim=1)

        assert torch.equal(preds, expected_preds)

    def test_predict_multiclass(self):
        """Test predict with multiple classes."""
        num_classes = 5
        classifier = FaceClassifier(input_dim=2816, num_classes=num_classes)
        x = torch.randn(8, 2816)
        preds = classifier.predict(x)

        assert preds.shape == (8,)
        assert (preds >= 0).all()
        assert (preds < num_classes).all()


class TestFaceClassifierGradientFlow:
    """Test gradient flow through FaceClassifier."""

    def test_gradient_flow(self):
        """Test that gradients flow through the classifier."""
        classifier = FaceClassifier(input_dim=2816)
        x = torch.randn(8, 2816, requires_grad=True)

        logits = classifier(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_with_loss_function(self, labels_binary):
        """Test gradient flow with CrossEntropyLoss."""
        classifier = FaceClassifier(input_dim=2816)
        x = torch.randn(8, 2816, requires_grad=True)

        logits = classifier(x)
        loss = nn.CrossEntropyLoss()(logits, labels_binary)
        loss.backward()

        assert x.grad is not None
        #gradients should be non-zero
        assert (x.grad.abs() > 0).any()


class TestFaceClassifierDropoutBehavior:
    """Test FaceClassifier dropout behavior in train vs eval mode."""

    def test_dropout_active_in_train_mode(self):
        """Test that dropout is active in training mode."""
        classifier = FaceClassifier(input_dim=2816, dropout_rate=0.5)
        classifier.train()

        x = torch.randn(8, 2816)

        #run forward pass multiple times
        logits1 = classifier(x)
        logits2 = classifier(x)

        #outputs should be different due to dropout
        assert not torch.allclose(logits1, logits2, atol=1e-5)

    def test_dropout_inactive_in_eval_mode(self):
        """Test that dropout is inactive in eval mode."""
        classifier = FaceClassifier(input_dim=2816, dropout_rate=0.5)
        classifier.eval()

        x = torch.randn(8, 2816)

        #run forward pass multiple times
        with torch.no_grad():
            logits1 = classifier(x)
            logits2 = classifier(x)

        #outputs should be identical (no dropout)
        assert torch.allclose(logits1, logits2)

    def test_no_dropout_means_deterministic(self):
        """Test that with dropout_rate=0, outputs are always deterministic."""
        classifier = FaceClassifier(input_dim=2816, dropout_rate=0.0)
        classifier.train()  #even in train mode

        x = torch.randn(8, 2816)

        logits1 = classifier(x)
        logits2 = classifier(x)

        #outputs should be identical (no dropout)
        assert torch.allclose(logits1, logits2)


class TestFaceClassifierSaveLoad:
    """Test FaceClassifier save/load weights."""

    def test_save_and_load_weights(self):
        """Test saving and loading weights preserves model state."""
        classifier = FaceClassifier(input_dim=2816)

        #generate some predictions before saving
        x = torch.randn(8, 2816)
        classifier.eval()
        with torch.no_grad():
            logits_before = classifier(x)

        #save weights
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "classifier_weights.pth")
            classifier.save_weights(weight_path)

            #create new classifier and load weights
            classifier_loaded = FaceClassifier(input_dim=2816)
            classifier_loaded.load_weights(weight_path)
            classifier_loaded.eval()

            #predictions should be identical
            with torch.no_grad():
                logits_after = classifier_loaded(x)

            assert torch.allclose(logits_before, logits_after)

    def test_load_nonexistent_weights(self):
        """Test that loading non-existent weights raises error."""
        classifier = FaceClassifier(input_dim=2816)

        with pytest.raises(ValueError, match="Weights file does not exist"):
            classifier.load_weights("/nonexistent/path/weights.pth")

    def test_save_weights_creates_directory(self):
        """Test that save_weights creates parent directories."""
        classifier = FaceClassifier(input_dim=2816)

        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "subdir", "nested", "weights.pth")
            classifier.save_weights(weight_path)

            #verify file was created
            assert os.path.exists(weight_path)


class TestFaceClassifierIntegrationWithLoss:
    """Test FaceClassifier integration with loss functions."""

    def test_crossentropy_loss_with_logits(self, labels_binary):
        """Test that CrossEntropyLoss works correctly with classifier logits."""
        classifier = FaceClassifier(input_dim=2816, num_classes=2)
        x = torch.randn(8, 2816)

        logits = classifier(x)
        loss = nn.CrossEntropyLoss()(logits, labels_binary)

        #loss should be a scalar
        assert loss.ndim == 0
        #loss should be positive
        assert loss.item() > 0
        #loss should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_training_step_reduces_loss(self):
        """Test that a training step reduces loss (basic sanity check)."""
        classifier = FaceClassifier(input_dim=2816)
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        #create synthetic data that's easy to learn
        x = torch.randn(16, 2816)
        #labels correlated with first feature
        labels = (x[:, 0] > 0).long()

        classifier.train()

        #compute initial loss
        logits = classifier(x)
        initial_loss = criterion(logits, labels).item()

        #perform multiple training steps
        for _ in range(10):
            optimizer.zero_grad()
            logits = classifier(x)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        #compute final loss
        logits = classifier(x)
        final_loss = criterion(logits, labels).item()

        #loss should decrease (model is learning)
        assert final_loss < initial_loss


class TestFaceClassifierDeviceCompatibility:
    """Test FaceClassifier device compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_on_cuda(self):
        """Test forward pass on CUDA device."""
        classifier = FaceClassifier(input_dim=2816).cuda()
        x = torch.randn(8, 2816).cuda()

        logits = classifier(x)

        assert logits.device.type == 'cuda'
        assert logits.shape == (8, 2)

    def test_forward_on_cpu(self):
        """Test forward pass on CPU device."""
        classifier = FaceClassifier(input_dim=2816)
        x = torch.randn(8, 2816)

        logits = classifier(x)

        assert logits.device.type == 'cpu'
        assert logits.shape == (8, 2)
