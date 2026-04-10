"""Utilities for creating train/val/test splits with optional group awareness and k-fold CV."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)


def create_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    group_ids: list[int] | None = None,
    group_key: str | None = None,
) -> tuple[Subset, Subset | None, Subset | None]:
    """Create deterministic train/val/test splits with optional group awareness.

    When group_ids are provided, entire groups (e.g., sequences, videos) are kept together
    in the same split rather than individual samples being split.

    Args:
        dataset: PyTorch dataset instance.
        train_ratio: Fraction of data for training split.
        val_ratio: Fraction of data for validation split.
        test_ratio: Fraction of data for test split.
        seed: Random seed for reproducibility.
        group_ids: Optional list of group IDs (one per sample) for group-aware splitting.
            If provided, samples with the same group_id are kept together.
        group_key: Name of the group key (for logging purposes).

    Returns:
        Tuple of (train_subset, val_subset, test_subset). val_subset or test_subset
        may be None if their ratio is 0.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    num_samples = len(dataset)
    generator = random.Random(seed)

    # Determine if we're doing group-aware split
    if group_ids is not None:
        logger.info(
            f"Using group-aware split with {group_key or 'group_id'}. "
            f"Grouping {len(set(group_ids))} unique groups."
        )
        unique_groups = sorted(set(group_ids))
        group_indices = {g: [] for g in unique_groups}
        for idx, gid in enumerate(group_ids):
            group_indices[gid].append(idx)

        # Shuffle groups, not individual samples
        groups_shuffled = unique_groups.copy()
        generator.shuffle(groups_shuffled)

        # Distribute groups by ratio
        num_groups = len(groups_shuffled)
        val_group_count = max(1, int(num_groups * val_ratio)) if val_ratio > 0 else 0
        test_group_count = max(1, int(num_groups * test_ratio)) if test_ratio > 0 else 0
        train_group_count = num_groups - val_group_count - test_group_count

        train_groups = groups_shuffled[:train_group_count]
        val_groups = groups_shuffled[
            train_group_count : train_group_count + val_group_count
        ]
        test_groups = groups_shuffled[train_group_count + val_group_count :]

        # Flatten group indices to sample indices
        train_indices = []
        for g in train_groups:
            train_indices.extend(group_indices[g])
        val_indices = []
        for g in val_groups:
            val_indices.extend(group_indices[g])
        test_indices = []
        for g in test_groups:
            test_indices.extend(group_indices[g])

        logger.info(
            f"Split: train={len(train_indices)} samples ({len(train_groups)} groups), "
            f"val={len(val_indices)} samples ({len(val_groups)} groups), "
            f"test={len(test_indices)} samples ({len(test_groups)} groups)"
        )
    else:
        # Simple random split
        indices = list(range(num_samples))
        generator.shuffle(indices)

        val_count = int(num_samples * val_ratio)
        test_count = int(num_samples * test_ratio)
        train_count = num_samples - val_count - test_count

        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        test_indices = indices[train_count + val_count :]

        logger.info(
            f"Split: train={len(train_indices)} samples, "
            f"val={len(val_indices)} samples, test={len(test_indices)} samples"
        )

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if len(val_indices) > 0 else None
    test_subset = Subset(dataset, test_indices) if len(test_indices) > 0 else None

    return train_subset, val_subset, test_subset


def create_kfold_splits(
    dataset: Dataset,
    num_folds: int = 5,
    fold_index: int = 0,
    seed: int = 42,
    group_ids: list[int] | None = None,
    group_key: str | None = None,
) -> tuple[Subset, Subset]:
    """Create k-fold cross-validation splits.

    Returns train and validation subsets for the given fold.

    Args:
        dataset: PyTorch dataset instance.
        num_folds: Total number of folds.
        fold_index: Index of the current fold (0-based).
        seed: Random seed for reproducibility.
        group_ids: Optional list of group IDs for group-aware folding.
        group_key: Name of the group key (for logging purposes).

    Returns:
        Tuple of (train_subset, val_subset) for the current fold.
    """
    assert 0 <= fold_index < num_folds, f"fold_index must be in [0, {num_folds})"

    num_samples = len(dataset)
    generator = random.Random(seed)

    if group_ids is not None:
        logger.info(
            f"Using group-aware k-fold split (fold {fold_index}/{num_folds}) "
            f"with {group_key or 'group_id'}."
        )
        unique_groups = sorted(set(group_ids))
        group_indices = {g: [] for g in unique_groups}
        for idx, gid in enumerate(group_ids):
            group_indices[gid].append(idx)

        groups_shuffled = unique_groups.copy()
        generator.shuffle(groups_shuffled)

        fold_size = len(groups_shuffled) // num_folds
        val_start_group = fold_index * fold_size
        val_end_group = (
            (fold_index + 1) * fold_size
            if fold_index < num_folds - 1
            else len(groups_shuffled)
        )

        val_groups = groups_shuffled[val_start_group:val_end_group]
        train_groups = (
            groups_shuffled[:val_start_group] + groups_shuffled[val_end_group:]
        )

        train_indices = []
        for g in train_groups:
            train_indices.extend(group_indices[g])
        val_indices = []
        for g in val_groups:
            val_indices.extend(group_indices[g])

        logger.info(
            f"Fold {fold_index}: train={len(train_indices)} samples ({len(train_groups)} groups), "
            f"val={len(val_indices)} samples ({len(val_groups)} groups)"
        )
    else:
        indices = list(range(num_samples))
        generator.shuffle(indices)

        fold_size = num_samples // num_folds
        val_start = fold_index * fold_size
        val_end = (
            (fold_index + 1) * fold_size if fold_index < num_folds - 1 else num_samples
        )

        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        logger.info(
            f"Fold {fold_index}: train={len(train_indices)} samples, val={len(val_indices)} samples"
        )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


def extract_group_ids_from_dataset(
    dataset: Dataset, group_key: str | None = None
) -> list[int] | None:
    """Attempt to extract group IDs from dataset samples.

    Looks for a group key in the sample dict returned by dataset[idx].
    Common keys: 'video_id', 'sequence_id', 'source_id', 'group_id'.

    Args:
        dataset: PyTorch dataset instance.
        group_key: Specific key to look for. If None, tries common keys.

    Returns:
        List of group IDs (one per sample) or None if not available.
    """
    if len(dataset) == 0:
        return None

    # Try to get the first sample to inspect structure
    try:
        sample = dataset[0]
        if not isinstance(sample, dict):
            return None
    except (IndexError, RuntimeError):
        return None

    # Determine which key to use
    keys_to_try = (
        [group_key]
        if group_key
        else ["video_id", "sequence_id", "source_id", "group_id"]
    )

    for key in keys_to_try:
        if key in sample:
            try:
                group_ids = []
                for idx in range(len(dataset)):
                    s = dataset[idx]
                    if key in s:
                        gid = s[key]
                        if isinstance(gid, torch.Tensor):
                            gid = int(gid.item())
                        group_ids.append(int(gid))
                    else:
                        return None  # Key not consistently available
                return group_ids
            except (ValueError, TypeError, KeyError):
                continue

    logger.warning(
        f"Could not extract group IDs from dataset. "
        f"Tried keys: {keys_to_try}. Using simple random split."
    )
    return None
