from __future__ import annotations

import numpy as np
import ants


def same_grid(image: ants.ANTsImage, reference: ants.ANTsImage, tolerance: float = 1e-5) -> bool:
    return (
        tuple(image.shape) == tuple(reference.shape)
        and np.allclose(image.spacing, reference.spacing, atol=tolerance)
        and np.allclose(image.origin, reference.origin, atol=tolerance)
        and np.allclose(np.asarray(image.direction), np.asarray(reference.direction), atol=tolerance)
    )


def get_discrete_labels(image: ants.ANTsImage, name: str) -> list[int]:
    array = image.numpy()

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    values = np.unique(array)
    nonzero_values = values[~np.isclose(values, 0)]

    if nonzero_values.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.allclose(nonzero_values, np.round(nonzero_values), atol=1e-6):
        raise ValueError(
            f"{name} contains non-integer values:\n{nonzero_values[:20]}\n"
            "It may be a probabilistic image, so GenericLabel would not be appropriate."
        )

    return [int(round(value)) for value in nonzero_values]
