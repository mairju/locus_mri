from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import ants
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .utils import same_grid


ReferenceStatistic = Literal["mode", "mean", "median"]

PER_SLICE_DISPLAY_COLUMNS = [
    "z_index",
    "z_mni_mm",
    "reference_value",
    "left_peak_contrast",
    "right_peak_contrast",
    "mean_lr_peak_contrast",
    "left_cluster_contrast",
    "right_cluster_contrast",
    "mean_lr_cluster_contrast",
    "left_search_voxels",
    "right_search_voxels",
    "left_lc_voxels",
    "right_lc_voxels",
    "left_line_residual_mm",
    "right_line_residual_mm",
    "left_line_outlier",
    "right_line_outlier",
]


def _display_dataframe(dataframe: pd.DataFrame) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(dataframe)
            return
    except (ImportError, NameError):
        pass

    print(dataframe.to_string(index=False))


def display_per_slice_table(
    per_slice_df: pd.DataFrame,
    subject: str,
    session: str,
) -> pd.DataFrame:
    """Display slices for which a bilateral FT peak metric was obtained."""

    columns = [
        column
        for column in PER_SLICE_DISPLAY_COLUMNS
        if column in per_slice_df.columns
    ]

    valid_slice_table = (
        per_slice_df[columns]
        .dropna(subset=["mean_lr_peak_contrast"])
        .reset_index(drop=True)
    )

    print(f"\nPer-slice FT LC contrast: {subject}/{session}")
    print(f"Valid bilateral slices: {len(valid_slice_table)}")
    _display_dataframe(valid_slice_table)

    return valid_slice_table


def _mni_hemisphere_half_masks(
    image: ants.ANTsImage,
    midline_x_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3-D boolean masks for MNI x < midline and x > midline."""

    direction = np.asarray(image.direction, dtype=float)

    if not np.allclose(
        direction,
        np.diag(np.diag(direction)),
        atol=1e-5,
    ):
        raise ValueError(
            "The image grid is oblique. Automatic left/right splitting "
            "requires an axis-aligned MNI grid."
        )

    x_indices = np.arange(image.shape[0], dtype=float)
    x_coordinates_mm = (
        float(image.origin[0])
        + direction[0, 0]
        * x_indices
        * float(image.spacing[0])
    )

    left_half = np.broadcast_to(
        x_coordinates_mm[:, None, None] < midline_x_mm,
        image.shape,
    )
    right_half = np.broadcast_to(
        x_coordinates_mm[:, None, None] > midline_x_mm,
        image.shape,
    )

    return left_half, right_half


def split_bilateral_mask_by_mni_x(
    mask_image: ants.ANTsImage,
    midline_x_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a bilateral MNI-space mask into left and right search masks."""

    mask_array = mask_image.numpy() > 0
    left_half, right_half = _mni_hemisphere_half_masks(
        mask_image,
        midline_x_mm,
    )

    left_mask = mask_array & left_half
    right_mask = mask_array & right_half

    if not left_mask.any():
        raise RuntimeError(
            "The bilateral LC mask contains no voxels on the left side "
            f"of MNI x={midline_x_mm}."
        )

    if not right_mask.any():
        raise RuntimeError(
            "The bilateral LC mask contains no voxels on the right side "
            f"of MNI x={midline_x_mm}."
        )

    return left_mask, right_mask


def _dilate_search_mask(
    mask: np.ndarray,
    *,
    inplane_iterations: int,
    throughplane_iterations: int,
) -> np.ndarray:
    """Dilate separately in-plane and through-plane."""

    result = mask.astype(bool, copy=True)

    if inplane_iterations < 0 or throughplane_iterations < 0:
        raise ValueError("Dilation iterations cannot be negative.")

    if inplane_iterations:
        inplane_structure = np.zeros((3, 3, 3), dtype=bool)
        inplane_structure[:, :, 1] = True
        result = binary_dilation(
            result,
            structure=inplane_structure,
            iterations=inplane_iterations,
        )

    if throughplane_iterations:
        throughplane_structure = np.zeros((3, 3, 3), dtype=bool)
        throughplane_structure[1, 1, :] = True
        result = binary_dilation(
            result,
            structure=throughplane_structure,
            iterations=throughplane_iterations,
        )

    return result


def _estimate_reference_intensity(
    values: np.ndarray,
    statistic: ReferenceStatistic,
) -> float:
    """
    Estimate the same-slice reference intensity.

    The paper uses the mode. For interpolated floating-point MRI data,
    exact values are often unique, so the mode implementation first uses
    an exact mode when repeated values exist and otherwise uses a robust
    histogram-mode estimate.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    if statistic == "mean":
        return float(np.mean(values))

    if statistic == "median":
        return float(np.median(values))

    if statistic != "mode":
        raise ValueError(
            "reference_statistic must be 'mode', 'mean', or 'median'."
        )

    unique_values, counts = np.unique(values, return_counts=True)
    maximum_count = int(counts.max())

    if maximum_count > 1:
        return float(unique_values[np.argmax(counts)])

    if values.size < 8 or np.isclose(values.min(), values.max()):
        return float(np.median(values))

    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr = float(q75 - q25)

    if iqr > 0:
        bin_width = 2.0 * iqr / np.cbrt(values.size)
    else:
        bin_width = 0.0

    value_range = float(values.max() - values.min())

    if bin_width > 0:
        number_of_bins = int(np.ceil(value_range / bin_width))
    else:
        number_of_bins = int(np.ceil(np.sqrt(values.size)))

    number_of_bins = int(np.clip(number_of_bins, 8, 128))
    counts, edges = np.histogram(values, bins=number_of_bins)
    modal_bin = int(np.argmax(counts))

    lower = edges[modal_bin]
    upper = edges[modal_bin + 1]

    if modal_bin == number_of_bins - 1:
        in_modal_bin = values[(values >= lower) & (values <= upper)]
    else:
        in_modal_bin = values[(values >= lower) & (values < upper)]

    if in_modal_bin.size:
        return float(np.median(in_modal_bin))

    return float((lower + upper) / 2.0)


def _neighbor_offsets(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))

    if connectivity == 8:
        return tuple(
            (dx, dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        )

    raise ValueError("cluster_connectivity must be 4 or 8.")


def _best_connected_cluster(
    intensity_slice: np.ndarray,
    search_mask_slice: np.ndarray,
    number_of_voxels: int,
    *,
    connectivity: int,
) -> np.ndarray | None:
    """
    Find a high-mean connected cluster without an intensity threshold.

    For every possible seed voxel, a connected cluster is grown by adding
    the brightest available neighboring voxel until the requested size is
    reached. The cluster with the highest mean signal is retained.
    """

    if number_of_voxels < 1:
        raise ValueError("number_of_voxels must be at least 1.")

    valid_search = (
        search_mask_slice.astype(bool)
        & np.isfinite(intensity_slice)
    )

    coordinates = np.argwhere(valid_search)

    if coordinates.shape[0] < number_of_voxels:
        return None

    offsets = _neighbor_offsets(connectivity)
    shape_x, shape_y = intensity_slice.shape

    best_coordinates: set[tuple[int, int]] | None = None
    best_mean = -np.inf
    best_peak = -np.inf

    # Starting with brighter seeds first usually finds a strong solution early.
    seed_intensities = intensity_slice[
        coordinates[:, 0],
        coordinates[:, 1],
    ]
    seed_order = np.argsort(seed_intensities)[::-1]

    for seed_position in seed_order:
        seed = tuple(int(value) for value in coordinates[seed_position])
        selected: set[tuple[int, int]] = {seed}
        frontier: set[tuple[int, int]] = set()

        def add_neighbors(voxel: tuple[int, int]) -> None:
            x_index, y_index = voxel
            for dx, dy in offsets:
                neighbor = (x_index + dx, y_index + dy)
                nx, ny = neighbor
                if (
                    0 <= nx < shape_x
                    and 0 <= ny < shape_y
                    and valid_search[nx, ny]
                    and neighbor not in selected
                ):
                    frontier.add(neighbor)

        add_neighbors(seed)

        while len(selected) < number_of_voxels:
            if not frontier:
                break

            next_voxel = max(
                frontier,
                key=lambda voxel: float(intensity_slice[voxel]),
            )
            frontier.remove(next_voxel)
            selected.add(next_voxel)
            add_neighbors(next_voxel)
            frontier.difference_update(selected)

        if len(selected) != number_of_voxels:
            continue

        selected_list = list(selected)
        selected_values = np.asarray(
            [intensity_slice[voxel] for voxel in selected_list],
            dtype=float,
        )
        cluster_mean = float(selected_values.mean())
        cluster_peak = float(selected_values.max())

        if (
            cluster_mean > best_mean
            or (
                np.isclose(cluster_mean, best_mean)
                and cluster_peak > best_peak
            )
        ):
            best_mean = cluster_mean
            best_peak = cluster_peak
            best_coordinates = selected

    if best_coordinates is None:
        return None

    cluster_mask = np.zeros_like(valid_search, dtype=bool)
    for x_index, y_index in best_coordinates:
        cluster_mask[x_index, y_index] = True

    return cluster_mask


def _index_to_physical_point(
    image: ants.ANTsImage,
    index: tuple[int, int, int],
) -> np.ndarray:
    index_array = np.asarray(index, dtype=float)
    spacing = np.asarray(image.spacing, dtype=float)
    origin = np.asarray(image.origin, dtype=float)
    direction = np.asarray(image.direction, dtype=float)
    return origin + direction @ (index_array * spacing)


def _bilateral_mean(
    left_value: float,
    right_value: float,
    *,
    require_both_sides: bool,
) -> float:
    left_valid = np.isfinite(left_value)
    right_valid = np.isfinite(right_value)

    if left_valid and right_valid:
        return float(np.mean([left_value, right_value]))

    if require_both_sides:
        return float("nan")

    available = [
        value
        for value in (left_value, right_value)
        if np.isfinite(value)
    ]

    if not available:
        return float("nan")

    return float(np.mean(available))


def _add_linearity_qc(
    per_slice_df: pd.DataFrame,
    *,
    side: Literal["left", "right"],
    outlier_threshold_mm: float,
) -> None:
    """
    Fit linear trajectories x(z) and y(z) for LC peak coordinates.

    For each valid slice, calculate the radial distance between the observed
    LC peak and its position predicted by the fitted trajectory:

        residual = sqrt(
            (x_observed - x_fitted)**2
            + (y_observed - y_fitted)**2
        )

    The function modifies ``per_slice_df`` in place.

    At least three valid slices and at least two distinct z coordinates are
    required to fit and evaluate the trajectory.
    """

    if side not in {"left", "right"}:
        raise ValueError(
            f"side must be 'left' or 'right', but received {side!r}"
        )

    if not np.isfinite(outlier_threshold_mm) or outlier_threshold_mm < 0:
        raise ValueError(
            "outlier_threshold_mm must be a finite, non-negative number, "
            f"but received {outlier_threshold_mm!r}"
        )

    x_column = f"{side}_peak_x_mm"
    y_column = f"{side}_peak_y_mm"
    residual_column = f"{side}_line_residual_mm"
    outlier_column = f"{side}_line_outlier"

    required_columns = {
        x_column,
        y_column,
        "z_mni_mm",
    }

    missing_columns = required_columns.difference(per_slice_df.columns)

    if missing_columns:
        raise KeyError(
            "Missing columns required for linearity QC: "
            f"{sorted(missing_columns)}"
        )

    # Initialize output columns. They remain NaN for slices that cannot
    # participate in the trajectory fit.
    per_slice_df[residual_column] = np.nan
    per_slice_df[outlier_column] = np.nan

    x_numeric = pd.to_numeric(
        per_slice_df[x_column],
        errors="coerce",
    )
    y_numeric = pd.to_numeric(
        per_slice_df[y_column],
        errors="coerce",
    )
    z_numeric = pd.to_numeric(
        per_slice_df["z_mni_mm"],
        errors="coerce",
    )

    valid = (
        np.isfinite(x_numeric.to_numpy(dtype=float))
        & np.isfinite(y_numeric.to_numpy(dtype=float))
        & np.isfinite(z_numeric.to_numpy(dtype=float))
    )

    valid_indices = per_slice_df.index[valid]

    # Three slices provide at least one degree of freedom beyond the
    # two parameters of a first-degree polynomial.
    if len(valid_indices) < 3:
        return

    z_values = z_numeric.loc[valid_indices].to_numpy(dtype=float)
    x_values = x_numeric.loc[valid_indices].to_numpy(dtype=float)
    y_values = y_numeric.loc[valid_indices].to_numpy(dtype=float)

    # A line cannot be fitted when every observation has the same z value.
    if np.unique(z_values).size < 2:
        return

    try:
        x_coefficients = np.polyfit(
            z_values,
            x_values,
            deg=1,
        )
        y_coefficients = np.polyfit(
            z_values,
            y_values,
            deg=1,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        # Leave the QC columns as NaN if fitting is numerically impossible.
        return

    x_fitted = np.polyval(x_coefficients, z_values)
    y_fitted = np.polyval(y_coefficients, z_values)

    residuals = np.hypot(
        x_values - x_fitted,
        y_values - y_fitted,
    )

    finite_residuals = np.isfinite(residuals)

    if not np.any(finite_residuals):
        return

    fitted_indices = valid_indices[finite_residuals]
    fitted_residuals = residuals[finite_residuals]

    per_slice_df.loc[
        fitted_indices,
        residual_column,
    ] = fitted_residuals

    per_slice_df.loc[
        fitted_indices,
        outlier_column,
    ] = (
        fitted_residuals > outlier_threshold_mm
    ).astype(int)


def _new_image_like(
    reference_image: ants.ANTsImage,
    array: np.ndarray,
) -> ants.ANTsImage:
    return ants.from_numpy(
        array,
        origin=reference_image.origin,
        spacing=reference_image.spacing,
        direction=reference_image.direction,
    )


def _finite_mean(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def extract_literature_style_peak_contrast(
    subject,
    session,
    tse_in_mni_path,
    lc_mask_grid_path,
    dpt_mask_grid_path,
    output_dir,
    *,
    require_both_sides: bool = True,
    exclude_zero_tse: bool = True,
    show_per_slice: bool = True,
    overwrite: bool = False,
    search_dilation_inplane_voxels: int = 2,
    search_dilation_throughplane_voxels: int = 0,
    target_cluster_area_mm2: float = 1.96, # 1.96/(0.5x0.5) = 8
    cluster_voxels: int | None = None,
    cluster_connectivity: int = 8,
    reference_statistic: ReferenceStatistic = "mode",
    midline_x_mm: float = 0.0,
    linearity_outlier_threshold_mm: float | None = None,
):
    """
    FT-like not exactly the way they do

    Waht we are doing is treat lc atlas as a search mask
    1. dilate it and proccess left and right lcs separately




    REference:
    Characterization of an automated method to segment the human locus coeruleus, Sibahi1 et al. (2022)
    """

    subject = str(subject)
    session = str(session)

    tse_in_mni_path = Path(tse_in_mni_path)
    lc_mask_grid_path = Path(lc_mask_grid_path)
    dpt_mask_grid_path = Path(dpt_mask_grid_path)
    output_dir = Path(output_dir)

    required_inputs = {
        "TSE in MNI": tse_in_mni_path,
        "LC search mask": lc_mask_grid_path,
        "DPT/reference mask": dpt_mask_grid_path,
    }

    for name, path in required_inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_slice_path = output_dir / "lc_ft_contrast_per_slice.csv"
    summary_path = output_dir / "lc_ft_summary.json"
    search_mask_path = output_dir / "lc_ft_search_mask.nii.gz"
    cluster_mask_path = output_dir / "lc_ft_cluster_mask.nii.gz"
    peak_mask_path = output_dir / "lc_ft_peak_mask.nii.gz"

    output_paths = [
        per_slice_path,
        summary_path,
        search_mask_path,
        cluster_mask_path,
        peak_mask_path,
    ]

    if all(path.exists() for path in output_paths) and not overwrite:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

        per_slice_df = pd.read_csv(per_slice_path)

        print(f"\nReusing existing FT Step 8 results: {subject}/{session}")

        if show_per_slice:
            display_per_slice_table(
                per_slice_df=per_slice_df,
                subject=subject,
                session=session,
            )

        return summary, per_slice_df

    tse_image = ants.image_read(str(tse_in_mni_path)).clone("float")
    lc_image = ants.image_read(str(lc_mask_grid_path))
    dpt_image = ants.image_read(str(dpt_mask_grid_path))

    if not same_grid(lc_image, tse_image):
        raise RuntimeError("LC search mask does not share the TSE grid.")

    if not same_grid(dpt_image, tse_image):
        raise RuntimeError("DPT/reference mask does not share the TSE grid.")

    tse = tse_image.numpy().astype(np.float64)
    dpt_mask = dpt_image.numpy() > 0

    left_search_mask, right_search_mask = split_bilateral_mask_by_mni_x(
        lc_image,
        midline_x_mm=midline_x_mm,
    )

    left_half, right_half = _mni_hemisphere_half_masks(
        lc_image,
        midline_x_mm,
    )

    # Dilation is deliberately performed after the hemisphere split so the
    # left and right search masks cannot merge across the midline.
    left_search_mask = _dilate_search_mask(
        left_search_mask,
        inplane_iterations=search_dilation_inplane_voxels,
        throughplane_iterations=search_dilation_throughplane_voxels,
    ) & left_half

    right_search_mask = _dilate_search_mask(
        right_search_mask,
        inplane_iterations=search_dilation_inplane_voxels,
        throughplane_iterations=search_dilation_throughplane_voxels,
    ) & right_half

    valid_tse = np.isfinite(tse)

    if exclude_zero_tse:
        valid_tse &= ~np.isclose(tse, 0.0)

    inplane_voxel_area_mm2 = float(
        tse_image.spacing[0] * tse_image.spacing[1]
    )

    if cluster_voxels is None:
        if target_cluster_area_mm2 <= 0:
            raise ValueError("target_cluster_area_mm2 must be positive.")
        selected_cluster_voxels = max(
            1,
            int(round(target_cluster_area_mm2 / inplane_voxel_area_mm2)),
        )
    else:
        if cluster_voxels < 1:
            raise ValueError("cluster_voxels must be at least 1.")
        selected_cluster_voxels = int(cluster_voxels)

    actual_cluster_area_mm2 = (
        selected_cluster_voxels * inplane_voxel_area_mm2
    )

    if linearity_outlier_threshold_mm is None:
        linearity_outlier_threshold_mm = float(
            max(tse_image.spacing[0], tse_image.spacing[1])
        )

    selected_search_labels = np.zeros(tse.shape, dtype=np.uint8)
    selected_search_labels[left_search_mask] = 1
    selected_search_labels[right_search_mask] = 2

    selected_cluster_labels = np.zeros(tse.shape, dtype=np.uint8)
    selected_peak_labels = np.zeros(tse.shape, dtype=np.uint8)

    slice_rows: list[dict] = []

    for z_index in range(tse.shape[2]):
        tse_slice = tse[:, :, z_index]
        valid_slice = valid_tse[:, :, z_index]

        dpt_slice = dpt_mask[:, :, z_index] & valid_slice
        left_slice_search = left_search_mask[:, :, z_index] & valid_slice
        right_slice_search = right_search_mask[:, :, z_index] & valid_slice

        dpt_voxels = int(dpt_slice.sum())
        left_search_voxels = int(left_slice_search.sum())
        right_search_voxels = int(right_slice_search.sum())

        z_point = _index_to_physical_point(
            tse_image,
            (0, 0, z_index),
        )
        z_mm = float(z_point[2])

        reference_value = float("nan")
        if dpt_voxels > 0:
            reference_value = _estimate_reference_intensity(
                tse_slice[dpt_slice],
                reference_statistic,
            )

        reference_is_valid = (
            np.isfinite(reference_value)
            and not np.isclose(reference_value, 0.0)
        )

        row: dict = {
            "subject": subject,
            "session": session,
            "z_index": int(z_index),
            "z_mni_mm": z_mm,
            "reference_value": reference_value,
            "reference_statistic": reference_statistic,
            # Backward-compatible reference column name.
            "dpt_mean": reference_value,
            "dpt_voxels": dpt_voxels,
            "left_search_voxels": left_search_voxels,
            "right_search_voxels": right_search_voxels,
        }

        side_results: dict[str, dict] = {}

        for side, search_slice, label_value in (
            ("left", left_slice_search, 1),
            ("right", right_slice_search, 2),
        ):
            cluster_mask = _best_connected_cluster(
                tse_slice,
                search_slice,
                selected_cluster_voxels,
                connectivity=cluster_connectivity,
            )

            result = {
                "cluster_voxels": 0,
                "cluster_mean": float("nan"),
                "cluster_contrast": float("nan"),
                "peak_intensity": float("nan"),
                "peak_contrast": float("nan"),
                "peak_x_index": float("nan"),
                "peak_y_index": float("nan"),
                "peak_x_mm": float("nan"),
                "peak_y_mm": float("nan"),
            }

            if cluster_mask is not None:
                cluster_coordinates = np.argwhere(cluster_mask)
                cluster_values = tse_slice[cluster_mask]
                cluster_mean = float(cluster_values.mean())

                peak_position_in_cluster = int(np.argmax(cluster_values))
                peak_x_index, peak_y_index = (
                    int(value)
                    for value in cluster_coordinates[
                        peak_position_in_cluster
                    ]
                )
                peak_intensity = float(
                    tse_slice[peak_x_index, peak_y_index]
                )

                selected_cluster_labels[:, :, z_index][cluster_mask] = (
                    label_value
                )
                selected_peak_labels[
                    peak_x_index,
                    peak_y_index,
                    z_index,
                ] = label_value

                peak_point = _index_to_physical_point(
                    tse_image,
                    (peak_x_index, peak_y_index, z_index),
                )

                cluster_contrast = float("nan")
                peak_contrast = float("nan")

                if reference_is_valid:
                    cluster_contrast = float(
                        (cluster_mean - reference_value) / reference_value
                    )
                    peak_contrast = float(
                        (peak_intensity - reference_value) / reference_value
                    )

                result = {
                    "cluster_voxels": int(cluster_mask.sum()),
                    "cluster_mean": cluster_mean,
                    "cluster_contrast": cluster_contrast,
                    "peak_intensity": peak_intensity,
                    "peak_contrast": peak_contrast,
                    "peak_x_index": peak_x_index,
                    "peak_y_index": peak_y_index,
                    "peak_x_mm": float(peak_point[0]),
                    "peak_y_mm": float(peak_point[1]),
                }

            side_results[side] = result

            row.update(
                {
                    f"{side}_lc_voxels": result["cluster_voxels"],
                    f"{side}_cluster_mean": result["cluster_mean"],
                    f"{side}_cluster_contrast": result[
                        "cluster_contrast"
                    ],
                    f"{side}_peak_intensity": result[
                        "peak_intensity"
                    ],
                    f"{side}_peak_contrast": result["peak_contrast"],
                    f"{side}_peak_x_index": result["peak_x_index"],
                    f"{side}_peak_y_index": result["peak_y_index"],
                    f"{side}_peak_x_mm": result["peak_x_mm"],
                    f"{side}_peak_y_mm": result["peak_y_mm"],
                    # Backward-compatible LC mean column.
                    f"{side}_lc_mean": result["cluster_mean"],
                }
            )

        mean_lr_cluster_contrast = _bilateral_mean(
            side_results["left"]["cluster_contrast"],
            side_results["right"]["cluster_contrast"],
            require_both_sides=require_both_sides,
        )
        mean_lr_peak_contrast = _bilateral_mean(
            side_results["left"]["peak_contrast"],
            side_results["right"]["peak_contrast"],
            require_both_sides=require_both_sides,
        )

        row.update(
            {
                "mean_lr_cluster_contrast": mean_lr_cluster_contrast,
                "mean_lr_peak_contrast": mean_lr_peak_contrast,
                # Backward-compatible primary columns now use the FT
                # peak-intensity metric, which performed best in the paper.
                "left_contrast": side_results["left"]["peak_contrast"],
                "right_contrast": side_results["right"]["peak_contrast"],
                "mean_lr_contrast": mean_lr_peak_contrast,
            }
        )

        slice_rows.append(row)

    per_slice_df = pd.DataFrame(slice_rows)

    _add_linearity_qc(
        per_slice_df,
        side="left",
        outlier_threshold_mm=float(linearity_outlier_threshold_mm),
    )
    _add_linearity_qc(
        per_slice_df,
        side="right",
        outlier_threshold_mm=float(linearity_outlier_threshold_mm),
    )

    valid_left_peak = per_slice_df.dropna(subset=["left_peak_contrast"])
    valid_right_peak = per_slice_df.dropna(subset=["right_peak_contrast"])
    valid_bilateral_peak = per_slice_df.dropna(
        subset=["mean_lr_peak_contrast"]
    )
    valid_bilateral_cluster = per_slice_df.dropna(
        subset=["mean_lr_cluster_contrast"]
    )

    if valid_left_peak.empty:
        raise RuntimeError(
            "No valid left FT peak-voxel contrast values were calculated. "
            "Check whether the left search mask contains at least the "
            f"requested {selected_cluster_voxels} connected voxels per slice."
        )

    if valid_right_peak.empty:
        raise RuntimeError(
            "No valid right FT peak-voxel contrast values were calculated. "
            "Check whether the right search mask contains at least the "
            f"requested {selected_cluster_voxels} connected voxels per slice."
        )

    if valid_bilateral_peak.empty:
        raise RuntimeError(
            "No valid bilateral FT peak-voxel contrast values were calculated."
        )

    left_peak_row = valid_left_peak.loc[
        valid_left_peak["left_peak_contrast"].idxmax()
    ]
    right_peak_row = valid_right_peak.loc[
        valid_right_peak["right_peak_contrast"].idxmax()
    ]
    bilateral_peak_row = valid_bilateral_peak.loc[
        valid_bilateral_peak["mean_lr_peak_contrast"].idxmax()
    ]

    bilateral_cluster_peak_row = None
    if not valid_bilateral_cluster.empty:
        bilateral_cluster_peak_row = valid_bilateral_cluster.loc[
            valid_bilateral_cluster[
                "mean_lr_cluster_contrast"
            ].idxmax()
        ]

    left_line_valid = per_slice_df["left_line_outlier"].notna()
    right_line_valid = per_slice_df["right_line_outlier"].notna()

    left_line_outliers = int(
        per_slice_df.loc[left_line_valid, "left_line_outlier"].sum()
    )
    right_line_outliers = int(
        per_slice_df.loc[right_line_valid, "right_line_outlier"].sum()
    )

    summary = {
        "algorithm": "FT-like connected-cluster search",
        "algorithm_version": "step8_ft_v1",
        "subject": subject,
        "session": session,
        "space": "current TSE/MNI grid",
        "reference_statistic": reference_statistic,
        "contrast_formula": "(LC - same-slice reference) / reference",
        "search_dilation_inplane_voxels": int(
            search_dilation_inplane_voxels
        ),
        "search_dilation_throughplane_voxels": int(
            search_dilation_throughplane_voxels
        ),
        "cluster_connectivity": int(cluster_connectivity),
        "cluster_voxels_per_side_per_slice": int(
            selected_cluster_voxels
        ),
        "inplane_voxel_area_mm2": inplane_voxel_area_mm2,
        "target_cluster_area_mm2": float(target_cluster_area_mm2),
        "actual_cluster_area_mm2": float(actual_cluster_area_mm2),
        "ft_peak_voxel_peak_mean_lr_contrast": float(
            bilateral_peak_row["mean_lr_peak_contrast"]
        ),
        "ft_peak_voxel_peak_z": int(bilateral_peak_row["z_index"]),
        "ft_peak_voxel_peak_z_mni_mm": float(
            bilateral_peak_row["z_mni_mm"]
        ),
        "ft_peak_voxel_broad_mean_lr_contrast": _finite_mean(
            valid_bilateral_peak["mean_lr_peak_contrast"]
        ),
        "ft_cluster_broad_mean_lr_contrast": _finite_mean(
            valid_bilateral_cluster["mean_lr_cluster_contrast"]
        ),
        "left_peak_voxel_contrast": float(
            left_peak_row["left_peak_contrast"]
        ),
        "left_peak_voxel_z": int(left_peak_row["z_index"]),
        "left_peak_voxel_z_mni_mm": float(left_peak_row["z_mni_mm"]),
        "right_peak_voxel_contrast": float(
            right_peak_row["right_peak_contrast"]
        ),
        "right_peak_voxel_z": int(right_peak_row["z_index"]),
        "right_peak_voxel_z_mni_mm": float(
            right_peak_row["z_mni_mm"]
        ),
        "number_of_valid_bilateral_peak_slices": int(
            len(valid_bilateral_peak)
        ),
        "number_of_valid_bilateral_cluster_slices": int(
            len(valid_bilateral_cluster)
        ),
        "linearity_outlier_threshold_mm": float(
            linearity_outlier_threshold_mm
        ),
        "left_linearity_outliers": left_line_outliers,
        "left_linearity_valid_slices": int(left_line_valid.sum()),
        "right_linearity_outliers": right_line_outliers,
        "right_linearity_valid_slices": int(right_line_valid.sum()),
        "per_slice_csv": str(per_slice_path),
        "search_mask_path": str(search_mask_path),
        "cluster_mask_path": str(cluster_mask_path),
        "peak_mask_path": str(peak_mask_path),
        # Backward-compatible summary keys. These now refer to the FT
        # single-peak-voxel metric rather than the mean of the full atlas ROI.
        "literature_style_peak_mean_lr_contrast": float(
            bilateral_peak_row["mean_lr_peak_contrast"]
        ),
        "literature_style_peak_z": int(bilateral_peak_row["z_index"]),
        "literature_style_peak_z_mni_mm": float(
            bilateral_peak_row["z_mni_mm"]
        ),
        "left_peak_contrast": float(
            left_peak_row["left_peak_contrast"]
        ),
        "left_peak_z": int(left_peak_row["z_index"]),
        "left_peak_z_mni_mm": float(left_peak_row["z_mni_mm"]),
        "right_peak_contrast": float(
            right_peak_row["right_peak_contrast"]
        ),
        "right_peak_z": int(right_peak_row["z_index"]),
        "right_peak_z_mni_mm": float(right_peak_row["z_mni_mm"]),
        "number_of_valid_bilateral_slices": int(
            len(valid_bilateral_peak)
        ),
    }

    if bilateral_cluster_peak_row is not None:
        summary.update(
            {
                "ft_cluster_peak_mean_lr_contrast": float(
                    bilateral_cluster_peak_row[
                        "mean_lr_cluster_contrast"
                    ]
                ),
                "ft_cluster_peak_z": int(
                    bilateral_cluster_peak_row["z_index"]
                ),
                "ft_cluster_peak_z_mni_mm": float(
                    bilateral_cluster_peak_row["z_mni_mm"]
                ),
            }
        )

    per_slice_df.to_csv(per_slice_path, index=False)

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=True)

    ants.image_write(
        _new_image_like(tse_image, selected_search_labels),
        str(search_mask_path),
    )
    ants.image_write(
        _new_image_like(tse_image, selected_cluster_labels),
        str(cluster_mask_path),
    )
    ants.image_write(
        _new_image_like(tse_image, selected_peak_labels),
        str(peak_mask_path),
    )

    if show_per_slice:
        display_per_slice_table(
            per_slice_df=per_slice_df,
            subject=subject,
            session=session,
        )

    summary_columns = [
        "subject",
        "session",
        "ft_peak_voxel_peak_mean_lr_contrast",
        "ft_peak_voxel_peak_z",
        "ft_peak_voxel_broad_mean_lr_contrast",
        "ft_cluster_broad_mean_lr_contrast",
        "left_peak_voxel_contrast",
        "left_peak_voxel_z",
        "right_peak_voxel_contrast",
        "right_peak_voxel_z",
        "left_linearity_outliers",
        "right_linearity_outliers",
    ]

    summary_table = pd.DataFrame([summary])[summary_columns]

    print(f"\nFT LC contrast summary: {subject}/{session}")
    _display_dataframe(summary_table)

    print(f"\nSaved per-slice FT results: {per_slice_path}")
    print(f"Saved FT summary: {summary_path}")
    print(f"Saved dilated search mask: {search_mask_path}")
    print(f"Saved selected cluster mask: {cluster_mask_path}")
    print(f"Saved selected peak mask: {peak_mask_path}")

    return summary, per_slice_df
