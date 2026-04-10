"""Sampling strategies for low-data object detection with class imbalance."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler, Subset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class WeightedImageSampler(WeightedRandomSampler):
    """Image-level weighted sampler using inverse class frequency.

    Images containing rare classes receive higher sampling probability.
    Only applied to training set, not validation/test.
    """

    @staticmethod
    def compute_sample_weights(
        dataset: Dataset,
        class_freq_weight: bool = True,
        pow_factor: float = 0.5,
        max_weight_cap: float = 10.0,
    ) -> list[float]:
        """Compute per-image weights based on class distribution.

        Args:
            dataset: Detection dataset where samples have 'target' dict with 'labels'.
            class_freq_weight: If True, use inverse class frequency. If False, uniform.
            pow_factor: Power factor for weight computation. Lower values smooth weights.
            max_weight_cap: Maximum weight per image to avoid pathological oversampling.

        Returns:
            List of weights (one per sample).
        """
        weights = []

        # Collect class frequencies
        class_counts: dict[int, int] = {}
        total_samples = len(dataset)

        # Handle both Subset and direct Dataset
        if isinstance(dataset, Subset):
            actual_dataset = dataset.dataset
            indices = dataset.indices
        else:
            actual_dataset = dataset
            indices = list(range(len(dataset)))

        for idx in indices:
            try:
                if isinstance(dataset, Subset):
                    sample = dataset[idx]
                else:
                    sample = dataset[idx]
                target = sample.get("target")
                if target is None:
                    continue
                labels = target.get("labels")
                if labels is None:
                    continue
                labels_list = (
                    labels.tolist() if hasattr(labels, "tolist") else list(labels)
                )
                for label in labels_list:
                    class_counts[int(label)] = class_counts.get(int(label), 0) + 1
            except (IndexError, KeyError, TypeError):
                continue

        if not class_counts:
            logger.warning(
                "Could not compute class frequencies. Using uniform weights."
            )
            return [1.0] * len(indices)

        # Compute inverse class frequencies
        total_annotations = sum(class_counts.values())
        class_weights = {
            cls_id: (total_annotations / count) ** pow_factor
            for cls_id, count in class_counts.items()
        }

        # Assign per-image weights
        for idx in indices:
            try:
                if isinstance(dataset, Subset):
                    sample = dataset[idx]
                else:
                    sample = dataset[idx]
                target = sample.get("target")
                labels = target.get("labels")
                if labels is None:
                    weights.append(1.0)
                    continue

                labels_list = (
                    labels.tolist() if hasattr(labels, "tolist") else list(labels)
                )
                if not labels_list:
                    weights.append(1.0)
                    continue

                # Max weight of classes in this image
                image_weight = max(
                    class_weights.get(int(cls_id), 1.0) for cls_id in labels_list
                )
                image_weight = min(image_weight, max_weight_cap)
                weights.append(float(image_weight))
            except (IndexError, KeyError, TypeError):
                weights.append(1.0)

        logger.info(
            f"Computed weighted sampler with {len(weights)} samples, {len(class_counts)} classes"
        )
        return weights

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        num_samples: int | None = None,
        replacement: bool = True,
        class_freq_weight: bool = True,
        pow_factor: float = 0.5,
        max_weight_cap: float = 10.0,
    ) -> WeightedImageSampler:
        """Create a WeightedImageSampler from a dataset.

        Args:
            dataset: Detection dataset.
            num_samples: Number of samples to draw. If None, uses dataset length.
            replacement: Whether to sample with replacement.
            class_freq_weight: If True, use inverse class frequency weighting.
            pow_factor: Power factor for weight computation.
            max_weight_cap: Maximum weight per image.

        Returns:
            A WeightedImageSampler instance.
        """
        weights = cls.compute_sample_weights(
            dataset=dataset,
            class_freq_weight=class_freq_weight,
            pow_factor=pow_factor,
            max_weight_cap=max_weight_cap,
        )

        if num_samples is None:
            num_samples = len(weights)

        return cls(weights=weights, num_samples=num_samples, replacement=replacement)


class RepeatFactorSampler(Sampler):
    """Repeat-factor style sampler inspired by Detectron2.

    Underrepresented classes cause images containing them to be repeated more often.
    """

    def __init__(
        self,
        dataset: Dataset,
        repeat_factor_threshold: float = 0.001,
        seed: int = 0,
    ):
        """Initialize repeat-factor sampler.

        Args:
            dataset: Detection dataset where samples have 'target' dict with 'labels'.
            repeat_factor_threshold: Threshold for repeat factor computation.
                Classes with frequency < threshold get repeated.
            seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.repeat_factor_threshold = repeat_factor_threshold
        self.seed = seed
        self.repeat_indices = self._compute_repeat_indices()

    def _compute_repeat_indices(self) -> list[int]:
        """Compute repeated indices based on class frequencies.

        Returns:
            List of indices with repetition based on repeat factors.
        """
        # Handle Subset wrapper
        if isinstance(self.dataset, Subset):
            actual_dataset = self.dataset.dataset
            indices = self.dataset.indices
        else:
            actual_dataset = self.dataset
            indices = list(range(len(self.dataset)))

        # Collect class frequencies
        class_counts: dict[int, int] = {}
        image_to_classes: dict[int, set[int]] = {
            idx: set() for idx in range(len(indices))
        }

        for local_idx, sample_idx in enumerate(indices):
            try:
                if isinstance(self.dataset, Subset):
                    sample = self.dataset[local_idx]
                else:
                    sample = self.dataset[sample_idx]
                target = sample.get("target")
                if target is None:
                    continue
                labels = target.get("labels")
                if labels is None:
                    continue
                labels_list = (
                    labels.tolist() if hasattr(labels, "tolist") else list(labels)
                )
                for label in labels_list:
                    label_int = int(label)
                    class_counts[label_int] = class_counts.get(label_int, 0) + 1
                    image_to_classes[local_idx].update([label_int])
            except (IndexError, KeyError, TypeError):
                continue

        if not class_counts:
            logger.warning("Could not compute repeat factors. Returning indices as-is.")
            return list(range(len(self.dataset)))

        # Compute class frequencies
        total_count = sum(class_counts.values())
        class_freqs = {
            cls_id: count / total_count for cls_id, count in class_counts.items()
        }

        # Compute repeat factor per image
        repeat_indices = []
        for local_idx in range(len(indices)):
            classes_in_image = image_to_classes[local_idx]
            if not classes_in_image:
                repeat_indices.append(local_idx)
                continue

            # Repeat factor = max repeat factor among classes in image
            repeat_factors = []
            for cls_id in classes_in_image:
                freq = class_freqs.get(cls_id, 0.0)
                if freq > 0:
                    repeat_factor = max(
                        1.0, (self.repeat_factor_threshold / freq) ** 0.5
                    )
                    repeat_factors.append(repeat_factor)

            if repeat_factors:
                repeat_factor = max(repeat_factors)
                repeat_count = max(1, int(round(repeat_factor)))
                repeat_indices.extend([local_idx] * repeat_count)
            else:
                repeat_indices.append(local_idx)

        logger.info(
            f"Repeat-factor sampler: original size={len(indices)}, repeated size={len(repeat_indices)}, "
            f"repeat ratio={len(repeat_indices) / len(indices):.2f}x"
        )
        return repeat_indices

    def __iter__(self):
        """Yield shuffled repeated indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        shuffled = torch.randperm(len(self.repeat_indices), generator=g).tolist()
        return iter([self.repeat_indices[i] for i in shuffled])

    def __len__(self) -> int:
        """Return number of samples (with repetition)."""
        return len(self.repeat_indices)
