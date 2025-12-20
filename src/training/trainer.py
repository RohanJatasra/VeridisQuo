"""
Trainer module for DeepFakeDetector model training.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import wandb

from data.dataset import FaceDataset
from data.dataloader import DataLoaderWrapper
from core.model import DeepFakeDetector
from config import PROJECT_ROOT, MODEL_DIR



@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    train_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "train.csv")
    test_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "test.csv")
    eval_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "eval.csv")
    
@dataclass
class TestDatasetConfig:
    """Configuration for test datasets."""
    train_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "train.csv")
    test_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "test.csv")
    eval_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "eval.csv")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Device
    device: torch.device = field(default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    test: bool = False

    # Dataset paths (set in __post_init__ based on test flag)
    dataset_config: Optional[Union[DatasetConfig, TestDatasetConfig]] = None

    # Training hyperparameters
    batch_size: int = 64 # Optimal for GPU 16GB (use 16 if GPU 8GB)
    learning_rate: float = 1e-4  # Conservative LR to prevent NaN loss
    weight_decay: float = 1e-4  # Standard L2 regularization
    num_epochs: int = 50  # Full training
    num_workers: int = 8
    # utils
    compile_model: bool = False
    seed : int = 42
    use_automixed_precision: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1  # Evaluate every N epochs

    # Model saving
    save_dir: str = str(MODEL_DIR)
    save_best_only: bool = True
    model_name: str = "deepfake_detector"

    # Early stopping
    early_stopping_patience: int = 10

    # Wandb
    wandb_project: str = "deepfake-detector"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True

    def __post_init__(self):
        if self.dataset_config is None:
            self.dataset_config = TestDatasetConfig() if self.test else DatasetConfig()


class Trainer:
    """
    Trainer for DeepFakeDetector model.

    Handles training loop, validation, logging, and model checkpointing.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize Trainer.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger("training.Trainer")
        self.device = self.config.device

        # Initialize wandb
        self.wandb_run = None
        if self.config.use_wandb:
            self._init_wandb()
            
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.config.seed)
        
        torch.set_float32_matmul_precision('high')

        # Load datasets
        self.logger.info("Loading datasets...")
        self.train_dataset = FaceDataset(
            csv_path=self.config.dataset_config.train_csv,
            augment=True
        )
        self.test_dataset = FaceDataset(
            csv_path=self.config.dataset_config.test_csv,
            augment=False
        )
        self.eval_dataset = FaceDataset(
            csv_path=self.config.dataset_config.eval_csv,
            augment=False
        )

        # Create dataloaders
        self.train_loader = DataLoaderWrapper(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            device=self.device,
            drop_last=True
        )
        self.test_loader = DataLoaderWrapper(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            device=self.device
        )
        self.eval_loader = DataLoaderWrapper(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            device=self.device
        )

        # Initialize model
        self.logger.info("Initializing model...")
        self.model = DeepFakeDetector().to(self.device)
        
        if self.config.compile_model:
            self.logger.info("Compiling model for optimized performance...")
            self.model = torch.compile(self.model)
            self.logger.info("Model compilation complete.")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # GradScaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_automixed_precision and self.device.type == 'cuda' else None

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Trainer initialized:\n"
            f"  - Device: {self.device}\n"
            f"  - Train samples: {len(self.train_dataset)}\n"
            f"  - Test samples: {len(self.test_dataset)}\n"
            f"  - Eval samples: {len(self.eval_dataset)}\n"
            f"  - Batch size: {self.config.batch_size}\n"
            f"  - Learning rate: {self.config.learning_rate}\n"
            f"  - Epochs: {self.config.num_epochs}"
        )

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        run_name = f"veridisquo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.wandb_run = wandb.init(
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            name=run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "weight_decay": self.config.weight_decay,
                "architecture": "DeepFakeDetector",
                "optimizer": "AdamW",
            },
            reinit="finish_previous"
        )
        self.logger.info(f"Wandb initialized: {self.wandb_run.name}")

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics
        """
        self.logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()

                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)

                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])

                # Check for improvement
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0

                    if self.config.save_best_only:
                        self._save_model(is_best=True)
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break

            # Save checkpoint
            if not self.config.save_best_only:
                self._save_model(is_best=False)

        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self._evaluate(self.test_loader, "test")

        # Save final model
        self._save_model(is_best=False, final=True)

        if self.wandb_run:
            self.wandb_run.finish()

        self.logger.info(
            f"Training complete!\n"
            f"  - Best val loss: {self.best_val_loss:.4f}\n"
            f"  - Best val accuracy: {self.best_val_acc:.4f}\n"
            f"  - Test accuracy: {test_metrics['accuracy']:.4f}"
        )

        return test_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        # Accumulate tensors on GPU
        batch_losses = []
        batch_corrects = []
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=True
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            # Forward pass
            loss, correct = self._train_step(images, labels)

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"NaN or Inf loss detected at batch {batch_idx}!\n"
                    f"  - Loss value: {loss.item()}\n"
                    f"  - Images shape: {images.shape}\n"
                    f"  - Images min/max: {images.min().item():.4f}/{images.max().item():.4f}\n"
                    f"  - Labels: {labels}\n"
                )
                raise ValueError("Training stopped due to NaN/Inf loss")

            # Accumulate tensors (keep on GPU)
            batch_losses.append(loss * images.size(0))
            batch_corrects.append(correct)
            total_samples += images.size(0)

            # Update progress bar and log every 10 batches (periodic sync to CPU)
            if batch_idx % self.config.log_interval == 0:
                # Calculate running metrics from recent batches
                recent_start = max(0, len(batch_losses) - self.config.log_interval)
                recent_losses = torch.stack(batch_losses[recent_start:])
                recent_corrects = torch.stack(batch_corrects[recent_start:])
                recent_samples = min(total_samples, self.config.log_interval * images.size(0))

                running_loss = recent_losses.sum().item() / recent_samples
                running_acc = recent_corrects.sum().item() / recent_samples

                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'acc': f'{running_acc:.4f}'
                })

                # Log to wandb
                if self.wandb_run:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_acc": (correct.float() / images.size(0)).item(),
                        "global_step": self.global_step
                    })

            self.global_step += 1

        # End of epoch: single GPU aggregation then sync to CPU
        total_loss_tensor = torch.stack(batch_losses).sum()
        total_correct_tensor = torch.stack(batch_corrects).sum()

        return {
            'loss': (total_loss_tensor / total_samples).item(),
            'accuracy': (total_correct_tensor.float() / total_samples).item()
        }

    def _train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform single training step.

        Args:
            images: Batch of images [B, 3, 224, 224]
            labels: Batch of labels [B]

        Returns:
            Tuple of (loss tensor, correct count tensor)
        """
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.config.use_automixed_precision and self.scaler is not None:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate clipping)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

        # Calculate accuracy (keep as tensor)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum()

        return loss.detach(), correct

    def _validate(self) -> Dict[str, float]:
        """
        Run validation on eval set.

        Returns:
            Dictionary with validation metrics
        """
        return self._evaluate(self.eval_loader, "val")

    def _evaluate(
        self,
        dataloader: DataLoaderWrapper,
        split_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on given dataloader.

        Args:
            dataloader: DataLoader to evaluate on
            split_name: Name of split for logging

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_samples = 0

        # Accumulate tensors on GPU
        batch_losses = []
        all_predictions = []  # List of tensors
        all_labels = []       # List of tensors

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

                # Keep on GPU
                batch_losses.append(loss * images.size(0))
                all_predictions.append(predictions)
                all_labels.append(labels)
                total_samples += images.size(0)

        # Single GPU concatenation
        all_predictions_gpu = torch.cat(all_predictions, dim=0)
        all_labels_gpu = torch.cat(all_labels, dim=0)
        total_loss_tensor = torch.stack(batch_losses).sum()
        total_correct_tensor = (all_predictions_gpu == all_labels_gpu).sum()

        # Calculate basic metrics on GPU
        avg_loss = (total_loss_tensor / total_samples).item()
        accuracy = (total_correct_tensor.float() / total_samples).item()

        # Compute precision/recall/F1 on GPU
        metrics = self._compute_metrics_gpu(all_predictions_gpu, all_labels_gpu)
        metrics['loss'] = avg_loss
        metrics['accuracy'] = accuracy

        self.logger.info(
            f"{split_name.upper()} - Loss: {avg_loss:.4f}, "
            f"Acc: {accuracy:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

        return metrics

    def _compute_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            predictions: List of predicted labels
            labels: List of true labels

        Returns:
            Dictionary with precision, recall, F1
        """
        # For FAKE detection (class 0)
        tp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _compute_metrics_gpu(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score on GPU.

        Args:
            predictions (torch.Tensor): predicted labels [N]
            labels (torch.Tensor): true labels [N]

        Returns:
            Dict[str, float]: metrics dictionary
        """
        # Compute TP, FP, FN for class 0 (FAKE)
        tp = ((predictions == 0) & (labels == 0)).sum().float()
        fp = ((predictions == 0) & (labels == 1)).sum().float()
        fn = ((predictions == 1) & (labels == 0)).sum().float()

        # Compute metrics on GPU
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Single sync to CPU
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """Log metrics to wandb and console."""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if self.wandb_run:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })

    def _save_model(self, is_best: bool = False, final: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
            final: Whether this is the final model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        if is_best:
            path = Path(self.config.save_dir) / f"{self.config.model_name}_best.pth"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved best model to {path}")

        if final:
            path = Path(self.config.save_dir) / f"{self.config.model_name}_final.pth"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved final model to {path}")

        if not is_best and not final:
            path = Path(self.config.save_dir) / f"{self.config.model_name}_epoch{self.current_epoch + 1}.pth"
            torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']

        self.logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {self.current_epoch + 1})"
        )
