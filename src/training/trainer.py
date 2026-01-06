import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np

from data.dataset import FaceDataset
from data.dataloader import DataLoaderWrapper
from core import DeepFakeDetector
from utils import TrainingConfig

class Trainer:
    """Training orchestrator for DeepFakeDetector model.
    Manages complete training pipeline including data loading, optimization, validation,
    checkpointing, and experiment tracking via Weights & Biases. Implements mixed precision
    training, gradient clipping, learning rate scheduling (warmup + cosine annealing), and
    early stopping based on validation loss.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize Trainer with training configuration.
        Parameters:
            config (TrainingConfig): complete training configuration
        """
        self.logger = logging.getLogger("/".join(__name__.split("/")[-2:]))
        
        if config is None:
            error_msg = "Configuration is required. Please provide a TrainingConfig instance."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.config: TrainingConfig = config

        self.wandb_run: Optional[wandb.sdk.wandb_run.Run] = None
        """Weights & Biases run instance for experiment tracking."""
        self.current_epoch: int = 0
        """Current training epoch index."""
        self.global_step: int = 0
        """Global training step counter across all epochs."""
        self.best_val_loss: float = float('inf')
        """Best validation loss achieved during training."""
        self.best_val_acc: float = 0.0
        """Best validation accuracy achieved during training."""
        self.epochs_without_improvement: int = 0
        """Consecutive epochs without validation loss improvement for early stopping."""

        self._set_seed()
        torch.set_float32_matmul_precision('high')

        self.train_dataset: FaceDataset
        """Training dataset with augmentation enabled."""
        self.test_dataset: FaceDataset
        """Test dataset without augmentation."""
        self.eval_dataset: FaceDataset
        """Validation dataset without augmentation."""
        self.train_dataset, self.test_dataset, self.eval_dataset = self._load_datasets()

        self.train_loader: DataLoaderWrapper
        """Training dataloader with shuffling and drop_last enabled."""
        self.test_loader: DataLoaderWrapper
        """Test dataloader without shuffling."""
        self.eval_loader: DataLoaderWrapper
        """Validation dataloader without shuffling."""
        self.train_loader, self.test_loader, self.eval_loader = self._create_dataloaders()

        self.model: DeepFakeDetector = self._initialize_model()
        """DeepFakeDetector model instance on target device."""
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss().to(self.config.device)
        """Cross-entropy loss function for binary classification."""
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        """AdamW optimizer with weight decay."""

        self.scaler: Optional[torch.amp.GradScaler] = self._create_scaler()
        """Gradient scaler for automatic mixed precision training."""
        self.scheduler: LambdaLR = self._create_scheduler()
        """Learning rate scheduler with warmup and cosine annealing."""

        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        if self.config.use_wandb:
            self._init_wandb()

        self.logger.info(
            f"Trainer initialized:\n"
            f"  - Device: {self.config.device}\n"
            f"  - Train samples: {len(self.train_dataset)}\n"
            f"  - Test samples: {len(self.test_dataset)}\n"
            f"  - Eval samples: {len(self.eval_dataset)}\n"
            f"  - Batch size: {self.config.dataloaders_batch_size}\n"
            f"  - Learning rate: {self.config.learning_rate}\n"
            f"  - Epochs: {self.config.num_epochs}"
        )

    def _set_seed(self) -> None:
        """Configure random seed for reproducibility."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if self.config.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.config.seed)

    def _load_datasets(self) -> Tuple[FaceDataset, FaceDataset, FaceDataset]:
        """Load train, test, and eval datasets from CSV paths.
        Returns:
            Tuple[FaceDataset, FaceDataset, FaceDataset]: train, test, eval datasets
        """
        self.logger.info("Loading datasets...")
        train_dataset = FaceDataset(csv_path=self.config.train_csv_path, augment=True)
        test_dataset = FaceDataset(csv_path=self.config.test_csv_path, augment=False)
        eval_dataset = FaceDataset(csv_path=self.config.eval_csv_path, augment=False)
        return train_dataset, test_dataset, eval_dataset

    def _create_dataloaders(self) -> Tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]:
        """Create dataloaders for train, test, and eval datasets.
        Returns:
            Tuple[DataLoaderWrapper, DataLoaderWrapper, DataLoaderWrapper]: train, test, eval dataloaders
        """
        train_loader = DataLoaderWrapper(
            self.train_dataset,
            batch_size=self.config.dataloaders_batch_size,
            shuffle=True,
            num_workers=self.config.dataloaders_num_workers,
            device=self.config.device,
            drop_last=True
        )
        test_loader = DataLoaderWrapper(
            self.test_dataset,
            batch_size=self.config.dataloaders_batch_size,
            shuffle=False,
            num_workers=self.config.dataloaders_num_workers,
            device=self.config.device
        )
        eval_loader = DataLoaderWrapper(
            self.eval_dataset,
            batch_size=self.config.dataloaders_batch_size,
            shuffle=False,
            num_workers=self.config.dataloaders_num_workers,
            device=self.config.device
        )
        return train_loader, test_loader, eval_loader

    def _initialize_model(self) -> DeepFakeDetector:
        """Initialize and optionally compile DeepFakeDetector model.
        Returns:
            DeepFakeDetector: initialized model on target device
        """
        self.logger.info("Initializing model...")
        model = DeepFakeDetector().to(self.config.device)

        if self.config.compile_model:
            self.logger.info("Compiling model for optimized performance...")
            model = torch.compile(model)
            self.logger.info("Model compilation complete.")

        return model

    def _create_scaler(self) -> Optional[torch.amp.GradScaler]:
        """Create gradient scaler for mixed precision training if enabled.
        Returns:
            Optional[torch.amp.GradScaler]: scaler for mixed precision or None
        """
        if self.config.use_automixed_precision and self.config.device.type == 'cuda':
            return torch.amp.GradScaler('cuda')
        return None

    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup and cosine annealing.
        Returns:
            LambdaLR: learning rate scheduler
        """
        def lr_lambda(epoch):
            # warmup phase: linear increase from 0 to max_lr
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            # cosine annealing: smooth decrease from max_lr to min_lr
            else:
                progress = (epoch - self.config.warmup_epochs) / max(1, self.config.num_epochs - self.config.warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                lr_range = 1.0 - (self.config.min_learning_rate / self.config.learning_rate)
                return (self.config.min_learning_rate / self.config.learning_rate) + lr_range * cosine_decay

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging with current configuration."""
        run_name = f"veridisquo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.wandb_run = wandb.init(
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            name=run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "min_learning_rate": self.config.min_learning_rate,
                "batch_size": self.config.dataloaders_batch_size,
                "num_epochs": self.config.num_epochs,
                "warmup_epochs": self.config.warmup_epochs,
                "weight_decay": self.config.weight_decay,
                "architecture": "DeepFakeDetector",
                "optimizer": "AdamW",
                "scheduler": "Warmup + Cosine Annealing",
                "use_automixed_precision": self.config.use_automixed_precision,
            },
            reinit="finish_previous"
        )

        if self.config.log_gradients or self.config.log_weights:
            log_freq = max(100, len(self.train_loader) // 10)
            wandb.watch(
                self.model,
                log="all" if self.config.log_gradients and self.config.log_weights else "gradients" if self.config.log_gradients else "parameters",
                log_freq=log_freq
            )
        self.logger.info(f"Wandb initialized: {self.wandb_run.name}")

    def train(self) -> Dict[str, float]:
        """Execute complete training loop with evaluation and checkpointing.
        Returns:
            Dict[str, float]: final test metrics after training
        """
        self.logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch()

            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()
                self._log_metrics(train_metrics, val_metrics, epoch)

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                    self._save_model(is_best=True)
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            self.scheduler.step()
            self._save_model(is_latest=True)

        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self._evaluate(self.test_loader, "test")

        # Log final test metrics to wandb
        if self.wandb_run:
            wandb.log({
                "test/loss": test_metrics['loss'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall'],
                "test/f1": test_metrics['f1'],
            })
            # Log confusion matrix
            self._log_confusion_matrix(test_metrics)
            self.wandb_run.finish()

        self.logger.info(
            f"Training complete!\n"
            f"  - Best val loss: {self.best_val_loss:.4f}\n"
            f"  - Best val accuracy: {self.best_val_acc:.4f}\n"
            f"  - Test accuracy: {test_metrics['accuracy']:.4f}"
        )

        return test_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Execute one complete training epoch with progress tracking and logging.
        Returns:
            Dict[str, float]: epoch training metrics (loss, accuracy)
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

            # periodic logging and progress bar update
            if batch_idx % self.config.log_interval == 0:
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
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
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

    def _train_step(self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute single training step with forward pass, backpropagation, and optimization.
        Parameters:
            images (torch.Tensor): batch of images [B, 3, 224, 224]
            labels (torch.Tensor): batch of labels [B]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: loss tensor and correct prediction count
        """
        self.optimizer.zero_grad()

        if self.config.use_automixed_precision and self.scaler is not None:
            with torch.autocast(device_type=self.config.device.type, dtype=torch.float16):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        """Run validation on evaluation dataset.
        Returns:
            Dict[str, float]: validation metrics (loss, accuracy, precision, recall, f1)
        """
        return self._evaluate(self.eval_loader, "val")

    def _evaluate(self,
        dataloader: DataLoaderWrapper,
        split_name: str
    ) -> Dict[str, float]:
        """Evaluate model on specified dataset with comprehensive metrics computation.
        Parameters:
            dataloader (DataLoaderWrapper): dataloader for evaluation dataset
            split_name (str): dataset split name for logging (e.g., 'val', 'test')
        Returns:
            Dict[str, float]: evaluation metrics including loss, accuracy, precision, recall, f1,
                              per-class metrics, confusion matrix elements, and raw predictions/labels
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

        # Store predictions and labels for confusion matrix
        metrics['predictions'] = all_predictions_gpu
        metrics['labels'] = all_labels_gpu

        self.logger.info(
            f"{split_name.upper()} - Loss: {avg_loss:.4f}, "
            f"Acc: {accuracy:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

        return metrics

    def _compute_metrics_gpu(self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics on GPU for both FAKE and REAL classes.
        Parameters:
            predictions (torch.Tensor): predicted class labels [N]
            labels (torch.Tensor): ground truth class labels [N]
        Returns:
            Dict[str, float]: comprehensive metrics including precision, recall, f1 for both classes,
                              and confusion matrix elements (tp, fp, fn, tn) for FAKE class
        """
        # confusion matrix elements for both classes
        tp_fake = ((predictions == 0) & (labels == 0)).sum().float()
        fp_fake = ((predictions == 0) & (labels == 1)).sum().float()
        fn_fake = ((predictions == 1) & (labels == 0)).sum().float()
        tn_fake = ((predictions == 1) & (labels == 1)).sum().float()

        tp_real = ((predictions == 1) & (labels == 1)).sum().float()
        fp_real = ((predictions == 1) & (labels == 0)).sum().float()
        fn_real = ((predictions == 0) & (labels == 1)).sum().float()
        tn_real = ((predictions == 0) & (labels == 0)).sum().float()

        # metrics for FAKE class (primary detection task)
        precision = tp_fake / (tp_fake + fp_fake + 1e-10)
        recall = tp_fake / (tp_fake + fn_fake + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # metrics for REAL class
        precision_real = tp_real / (tp_real + fp_real + 1e-10)
        recall_real = tp_real / (tp_real + fn_real + 1e-10)
        f1_real = 2 * precision_real * recall_real / (precision_real + recall_real + 1e-10)

        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'precision_fake': precision.item(),
            'recall_fake': recall.item(),
            'f1_fake': f1.item(),
            'precision_real': precision_real.item(),
            'recall_real': recall_real.item(),
            'f1_real': f1_real.item(),
            'tp_fake': tp_fake.item(),
            'fp_fake': fp_fake.item(),
            'fn_fake': fn_fake.item(),
            'tn_fake': tn_fake.item(),
        }

    def _log_metrics(self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """Log training and validation metrics to console and wandb."""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if self.wandb_run:
            # Basic metrics
            metrics_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }

            # Per-class metrics
            if 'precision_fake' in val_metrics:
                metrics_dict.update({
                    "val/precision_fake": val_metrics['precision_fake'],
                    "val/recall_fake": val_metrics['recall_fake'],
                    "val/f1_fake": val_metrics['f1_fake'],
                    "val/precision_real": val_metrics['precision_real'],
                    "val/recall_real": val_metrics['recall_real'],
                    "val/f1_real": val_metrics['f1_real'],
                })

            # Confusion matrix elements
            if 'tp_fake' in val_metrics:
                metrics_dict.update({
                    "val/tp_fake": val_metrics['tp_fake'],
                    "val/fp_fake": val_metrics['fp_fake'],
                    "val/fn_fake": val_metrics['fn_fake'],
                    "val/tn_fake": val_metrics['tn_fake'],
                })

            wandb.log(metrics_dict)

            # Log confusion matrix visualization
            self._log_confusion_matrix(val_metrics, prefix="val")

    def _log_confusion_matrix(self, metrics: Dict, prefix: str = "test") -> None:
        """Log confusion matrix visualization to Weights & Biases.
        Parameters:
            metrics (Dict): metrics dictionary containing 'predictions' and 'labels' tensors
            prefix (str): logging prefix for wandb metric name (e.g., 'val', 'test')
        """
        if not self.wandb_run or 'predictions' not in metrics or 'labels' not in metrics:
            return

        # Move to CPU for wandb logging
        predictions = metrics['predictions'].cpu().numpy()
        labels = metrics['labels'].cpu().numpy()

        # Create confusion matrix
        wandb.log({
            f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predictions,
                class_names=["FAKE", "REAL"]
            )
        })

    def _save_model(self, is_best: bool = False, is_latest: bool = False) -> None:
        """Save model checkpoint with optimizer and scheduler state.
        Parameters:
            is_best (bool): whether this is the best model based on validation loss
            is_latest (bool): whether this is the most recent epoch checkpoint
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
            path = Path(self.config.save_dir) / "deepfake_detector_best.pth"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved BEST model to {path} (val_loss: {self.best_val_loss:.4f})")

        if is_latest:
            path = Path(self.config.save_dir) / "deepfake_detector_latest.pth"
            torch.save(checkpoint, path)
            if (self.current_epoch + 1) % 10 == 0:
                self.logger.info(f"Saved LATEST model to {path} (epoch {self.current_epoch + 1})")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model, optimizer, and scheduler state from checkpoint file.
        Parameters:
            checkpoint_path (str): path to .pth checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

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
