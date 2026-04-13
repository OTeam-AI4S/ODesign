"""Training module for gRNAde RNA inverse folding models.

This module provides the core training infrastructure for gRNAde, an SE(3)-equivariant
graph neural network for RNA sequence design conditioned on 3D structures. The training
pipeline includes:
    - Full training loop with validation and test evaluation
    - Learning rate scheduling based on self-consistency scores
    - Checkpoint management and model persistence
    - Integration with wandb for experiment tracking
    - Comprehensive evaluation metrics including recovery, perplexity, and self-consistency

Key Components:
    train: Main training loop orchestrating epochs, evaluation, and checkpointing
    loop: Single epoch training/evaluation over a data loader
    print_and_log: Unified logging to console and wandb
    print_confusion: Confusion matrix visualization for nucleotide predictions
"""

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import wandb
from src.constants import NUM_TO_LETTER
from src.evaluator import evaluate


##################################
# Main Training Function
##################################

def train(config, model, train_loader, val_loader, test_loader, device):
    """Train gRNAde RNA inverse folding model with comprehensive evaluation and checkpointing.

    This function implements the complete training pipeline for structure-conditioned RNA
    sequence design models. Training proceeds over multiple epochs with periodic validation,
    automatic checkpoint management, and comprehensive metric tracking via wandb.

    The training loop optimizes cross-entropy loss with optional label smoothing, using
    the Adam optimizer and ReduceLROnPlateau scheduler. Model performance is monitored
    via multiple metrics:
        - Recovery: Native sequence recovery rate
        - Perplexity: Sequence likelihood measure
        - Self-consistency: Agreement between designed sequences and target structures
          as evaluated by RibonanzaNet and RibonanzaNet-SS predictors

    The learning rate is adjusted based on validation self-consistency scores
    (sc_score_ribonanzanet_ss), and the best checkpoint is saved according to this metric.

    Args:
        config (dict): wandb configuration dictionary containing hyperparameters:
            - epochs (int): Number of training epochs
            - lr (float): Initial learning rate
            - label_smoothing (float): Label smoothing factor for training loss
            - val_every (int): Validation frequency (in epochs)
            - n_samples (int): Number of sequences to sample per structure for evaluation
            - temperature (float): Sampling temperature for sequence generation
            - save (bool): Whether to save model checkpoints
        model (torch.nn.Module): RNA inverse folding model (e.g., AutoregressiveMultiGNNv1)
        train_loader (torch.utils.data.DataLoader): Training data loader with batched RNA graphs
        val_loader (torch.utils.data.DataLoader): Validation data loader
        test_loader (torch.utils.data.DataLoader): Test data loader
        device (torch.device): Device for training (cpu, cuda)

    Workflow:
        1. Initialize optimizer (Adam), scheduler (ReduceLROnPlateau), and loss functions
        2. For each epoch:
            a. Train on training set with gradient updates
            b. Periodically evaluate on validation set
            c. If validation performance improves, evaluate on test set and save checkpoint
        3. After training, load best checkpoint and perform final evaluation on val and test sets
        4. Log all metrics to wandb and save results to disk
    """

    # Initialise loss function
    train_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    eval_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    # Initialise optimizer and scheduler
    lr = config.lr
    optimizer = Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=1, min_lr=0.00001)

    # Initialise lookup table mapping integers to nucleotides
    lookup = train_loader.dataset.featurizer.num_to_letter

    # Initialise best checkpoint information
    best_epoch, best_val_metric = -1, 0.0

    ##################################
    # Training loop over mini-batches
    ##################################

    for epoch in range(config.epochs):
        # Training iteration
        model.train()
        train_loss, train_acc, train_confusion = loop(
            model, train_loader, train_loss_fn, optimizer, device
        )
        print_and_log(
            epoch, train_loss, train_acc, train_confusion, lr=lr, mode="train", lookup=lookup
        )

        if epoch % config.val_every == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Evaluate on validation set
                val_loss, val_acc, val_confusion = loop(
                    model, val_loader, eval_loss_fn, None, device
                )
                results = evaluate(
                    model,
                    val_loader.dataset,
                    config.n_samples,
                    config.temperature,
                    device,
                    model_name="val",
                    metrics=[
                        "recovery",
                        "perplexity",
                        "sc_score_ribonanzanet",
                        "sc_score_ribonanzanet_ss",
                    ],
                )
                (
                    df,
                    samples_list,
                    recovery_list,
                    perplexity_list,
                    scscore_ribonanzanet_list,
                    scscore_ribonanzanet_ss_list,
                ) = results.values()
                val_metrics = {
                    "recovery": np.mean(recovery_list),
                    "perplexity": np.mean(perplexity_list),
                    "sc_score_ribonanzanet": np.mean(scscore_ribonanzanet_list),
                    "sc_score_ribonanzanet_ss": np.mean(scscore_ribonanzanet_ss_list),
                }
                print_and_log(
                    epoch,
                    val_loss,
                    val_acc,
                    val_confusion,
                    metrics=val_metrics,
                    mode="val",
                    lookup=lookup,
                )

                # LR scheduler step
                scheduler.step(val_metrics["sc_score_ribonanzanet_ss"])
                lr = optimizer.param_groups[0]["lr"]

                if val_metrics["sc_score_ribonanzanet_ss"] > best_val_metric:
                    # Update best checkpoint
                    best_epoch, best_val_loss, best_val_metric = (
                        epoch,
                        val_loss,
                        val_metrics["sc_score_ribonanzanet_ss"],
                    )

                    # Evaluate on test set
                    test_loss, test_acc, test_confusion = loop(
                        model, test_loader, eval_loss_fn, None, device
                    )
                    results = evaluate(
                        model,
                        test_loader.dataset,
                        config.n_samples,
                        config.temperature,
                        device,
                        model_name="test",
                        metrics=[
                            "recovery",
                            "perplexity",
                            "sc_score_ribonanzanet",
                            "sc_score_ribonanzanet_ss",
                        ],
                    )
                    (
                        df,
                        samples_list,
                        recovery_list,
                        perplexity_list,
                        scscore_ribonanzanet_list,
                        scscore_ribonanzanet_ss_list,
                    ) = results.values()
                    test_metrics = {
                        "recovery": np.mean(recovery_list),
                        "perplexity": np.mean(perplexity_list),
                        "sc_score_ribonanzanet": np.mean(scscore_ribonanzanet_list),
                        "sc_score_ribonanzanet_ss": np.mean(scscore_ribonanzanet_ss_list),
                    }
                    print_and_log(
                        epoch,
                        test_loss,
                        test_acc,
                        test_confusion,
                        metrics=test_metrics,
                        mode="test",
                        lookup=lookup,
                    )

                    if config.save:
                        # Save best checkpoint
                        checkpoint_path = os.path.join(wandb.run.dir, "best_checkpoint.h5")
                        torch.save(model.state_dict(), checkpoint_path)
                        wandb.run.summary["best_checkpoint"] = checkpoint_path
                        wandb.run.summary["best_epoch"] = best_epoch

        if config.save:
            # Save current epoch checkpoint
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "current_checkpoint.h5"))

    # End of training
    if config.save:
        # Evaluate best checkpoint
        print(
            f"EVALUATION: loading {os.path.join(wandb.run.dir, 'best_checkpoint.h5')} (epoch {best_epoch})"
        )
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, "best_checkpoint.h5")))

        for loader, set_name in [(test_loader, "test"), (val_loader, "val")]:
            # Run evaluator
            results = evaluate(
                model,
                loader.dataset,
                config.n_samples,
                config.temperature,
                device,
                model_name=set_name,
                metrics=[
                    "recovery",
                    "perplexity",
                    "sc_score_ribonanzanet",
                    "sc_score_ribonanzanet_ss",
                ],
                save_designs=False,
            )
            (
                df,
                samples_list,
                recovery_list,
                perplexity_list,
                scscore_ribonanzanet_list,
                scscore_ribonanzanet_ss_list,
            ) = results.values()
            # Save results
            torch.save(results, os.path.join(wandb.run.dir, f"{set_name}_results.pt"))
            # Update wandb summary metrics
            wandb.run.summary[f"best_{set_name}_recovery"] = np.mean(recovery_list)
            wandb.run.summary[f"best_{set_name}_perplexity"] = np.mean(perplexity_list)
            wandb.run.summary[f"best_{set_name}_scscore_ribonanzanet"] = np.mean(
                scscore_ribonanzanet_list
            )
            wandb.run.summary[f"best_{set_name}_scscore_ribonanzanet_ss"] = np.mean(
                scscore_ribonanzanet_ss_list
            )
            print(
                f"BEST {set_name} recovery: {np.mean(recovery_list):.4f}\
                    perplexity: {np.mean(perplexity_list):.4f}\
                    scscore_ribonanzanet: {np.mean(scscore_ribonanzanet_list):.4f}\
                    scscore_ribonanzanet_ss: {np.mean(scscore_ribonanzanet_ss_list):.4f}"
            )


##################################
# Training Loop Utilities
##################################

def loop(model, dataloader, loss_fn, optimizer=None, device="cpu"):
    """Execute a single training or evaluation epoch over the data loader.

    This function performs one complete pass through the provided data loader, computing
    loss, accuracy, and a confusion matrix for nucleotide prediction. It handles both
    training (with gradient updates) and evaluation (without updates) depending on whether
    an optimizer is provided.

    The function processes RNA structures as geometric graphs, where each nucleotide is
    a node with 3D coordinates and edges capture backbone connectivity, base pairing,
    and spatial proximity. The model predicts nucleotide identities (A, G, C, U) for
    positions indicated by the batch.mask_seq attribute.

    Args:
        model (torch.nn.Module): RNA inverse folding model that takes batched RNA graphs
            and returns logits of shape [num_nodes, num_nucleotide_types]
        dataloader (torch.utils.data.DataLoader): Data loader yielding batched RNA graphs.
            Each batch should have attributes:
            - seq: Ground truth nucleotide sequences (long tensor)
            - mask_seq: Boolean mask indicating positions to design
            - Additional graph attributes (coords, edge_index, etc.)
        loss_fn (torch.nn.Module): Loss function (typically CrossEntropyLoss) taking
            predicted logits and true labels
        optimizer (torch.optim.Optimizer, optional): Optimizer for parameter updates.
            If None, the function runs in evaluation mode without gradient computation
            or parameter updates. Defaults to None.
        device (str or torch.device): Device for computation ('cpu', 'cuda').
            Defaults to 'cpu'.

    Returns:
        tuple: A 3-tuple containing:
            - avg_loss (float): Average cross-entropy loss over all nucleotides
            - avg_accuracy (float): Fraction of correctly predicted nucleotides
            - confusion (np.ndarray): Confusion matrix of shape [num_classes, num_classes]
              where confusion[i, j] is the count of true class i predicted as class j

    Workflow:
        1. Initialize metrics (loss, accuracy, confusion matrix)
        2. For each batch:
            a. Move batch to specified device
            b. Forward pass through model to get logits
            c. Compute loss on masked positions
            d. If training, backpropagate loss and update parameters
            e. Update running metrics (loss, accuracy, confusion)
        3. Return averaged metrics over all batches
    """

    confusion = np.zeros((model.out_dim, model.out_dim))
    total_loss, total_correct, total_count = 0, 0, 0

    t = tqdm(dataloader)
    for batch in t:
        if optimizer:
            optimizer.zero_grad()

        # move batch to device
        batch = batch.to(device)

        try:
            logits = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise (e)
            print("Skipped batch due to OOM", flush=True)
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            continue

        # compute loss
        loss_value = loss_fn(logits[batch.mask_seq], batch.seq[batch.mask_seq])

        if optimizer:
            # backpropagate loss and update parameters
            loss_value.backward()
            optimizer.step()

        # update metrics
        num_nodes = int(batch.seq[batch.mask_seq].size(0))
        total_loss += float(loss_value.item()) * num_nodes
        total_count += num_nodes
        pred = torch.argmax(logits[batch.mask_seq], dim=-1).detach().cpu().numpy()
        true = batch.seq[batch.mask_seq].detach().cpu().numpy()
        total_correct += (pred == true).sum()
        confusion += confusion_matrix(true, pred, labels=range(model.out_dim))

        t.set_description(f"TRAIN loss: {total_loss / total_count:.4f}")

    return total_loss / total_count, total_correct / total_count, confusion


##################################
# Logging Utilities
##################################

def print_and_log(
    epoch,
    loss,
    acc,
    confusion,
    metrics=None,
    lr=None,
    mode="train",
    lookup=NUM_TO_LETTER,  # reverse of {'A': 0, 'G': 1, 'C': 2, 'U': 3}
):
    """Print training metrics to console and log them to wandb.

    This function provides unified logging for training, validation, and test metrics,
    combining console output with wandb tracking. It formats and displays loss, perplexity,
    accuracy, learning rate, and optional evaluation metrics, along with a confusion matrix
    showing per-nucleotide prediction performance.

    Args:
        epoch (int): Current training epoch number
        loss (float): Average cross-entropy loss for the epoch
        acc (float): Average accuracy (fraction of correct predictions) for the epoch
        confusion (np.ndarray): Confusion matrix of shape [num_classes, num_classes]
            showing prediction counts for each nucleotide type
        metrics (dict, optional): Additional evaluation metrics to log, where keys are
            metric names (e.g., 'recovery', 'sc_score_eternafold') and values are
            corresponding scores. Defaults to None.
        lr (float, optional): Current learning rate. If provided, will be logged to
            both console and wandb. Defaults to None.
        mode (str): Logging mode indicating the data split, typically 'train', 'val',
            or 'test'. Used as prefix for wandb metric names. Defaults to 'train'.
        lookup (dict): Mapping from integer nucleotide indices to letter codes
            (e.g., {0: 'A', 1: 'G', 2: 'C', 3: 'U'}). Used for confusion matrix
            labeling. Defaults to NUM_TO_LETTER.
    """
    # Create log string and wandb metrics dict
    log_str = (
        f"\nEPOCH {epoch} {mode.upper()} loss: {loss:.4f} perp: {np.exp(loss):.4f} acc: {acc:.4f}"
    )
    wandb_metrics = {
        f"{mode}/loss": loss,
        f"{mode}/perp": np.exp(loss),
        f"{mode}/acc": acc,
        "epoch": epoch,
    }

    if lr is not None:
        # Add learning rate to loggers
        log_str += f" lr: {lr:.6f}"
        wandb_metrics[f"lr"] = lr

    if metrics is not None:
        for metric_name, metric_value in metrics.items():
            # Add additional metrics to loggers
            log_str += f" {metric_name}: {metric_value:.4f}"
            wandb_metrics[f"{mode}/{metric_name}"] = metric_value

    print(log_str)
    print_confusion(confusion, lookup=lookup)
    wandb.log(wandb_metrics)


def print_confusion(mat, lookup):
    """Format and print confusion matrix for nucleotide predictions.

    Args:
        mat (np.ndarray): Raw confusion matrix of shape [num_classes, num_classes]
            where mat[i, j] represents the number of times true class i was predicted
            as class j. Typically num_classes = 4 for RNA (A, G, C, U).
        lookup (dict): Mapping from integer indices to nucleotide letter codes
            (e.g., {0: 'A', 1: 'G', 2: 'C', 3: 'U'}). Used to label rows and columns.
    """
    counts = mat.astype(np.int32)
    mat = (counts.T / counts.sum(axis=-1, keepdims=True).T).T
    mat = np.round(mat * 1000).astype(np.int32)
    res = "\n"
    for i in range(len(lookup.keys())):
        res += f"\t{lookup[i]}"
    res += "\tCount\n"
    for i in range(len(lookup.keys())):
        res += f"{lookup[i]}\t"
        res += "\t".join(f"{n}" for n in mat[i])
        res += f"\t{sum(counts[i])}\n"
    print(res)
