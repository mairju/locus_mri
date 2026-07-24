from __future__ import annotations

import json
from pathlib import Path

import ants
import numpy as np
import pandas as pd

from .utils import same_grid


PER_SLICE_DISPLAY_COLUMNS = [
    "z_index",
    "z_mni_mm",
    "left_contrast",
    "right_contrast",
    "mean_lr_contrast",
    "left_lc_voxels",
    "right_lc_voxels",
    "dpt_voxels",
]


def display_per_slice_table(
    per_slice_df: pd.DataFrame,
    subject: str,
    session: str,
) -> pd.DataFrame:

    valid_slice_table = (
        per_slice_df[
            PER_SLICE_DISPLAY_COLUMNS
        ]
        .dropna(
            subset=["mean_lr_contrast"]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nPer-slice LC contrast: "
        f"{subject}/{session}"
    )
    print(
        f"Valid bilateral slices: "
        f"{len(valid_slice_table)}"
    )

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(valid_slice_table)
        else:
            print(
                valid_slice_table.to_string(
                    index=False
                )
            )

    except (ImportError, NameError):
        print(
            valid_slice_table.to_string(
                index=False
            )
        )

    return valid_slice_table


def split_bilateral_mask_by_mni_x(
    mask_image: ants.ANTsImage,
    midline_x_mm: float = 0.0,
):

    mask_array = (
        mask_image.numpy() > 0
    )

    direction = np.asarray(
        mask_image.direction
    )

    if not np.allclose(
        direction,
        np.diag(np.diag(direction)),
        atol=1e-5,
    ):
        raise ValueError(
            "The mask grid is oblique. Automatic hemisphere "
            "splitting requires an axis-aligned MNI grid."
        )

    x_indices = np.arange(
        mask_image.shape[0]
    )

    x_coordinates_mm = (
        mask_image.origin[0]
        + direction[0, 0]
        * x_indices
        * mask_image.spacing[0]
    )

    left_half = (
        x_coordinates_mm[:, None, None]
        < midline_x_mm
    )

    right_half = (
        x_coordinates_mm[:, None, None]
        > midline_x_mm
    )

    left_mask = (
        mask_array
        & left_half
    )

    right_mask = (
        mask_array
        & right_half
    )

    if not left_mask.any():
        raise RuntimeError(
            "The bilateral LC mask contains no voxels on "
            "the left side of MNI x=0."
        )

    if not right_mask.any():
        raise RuntimeError(
            "The bilateral LC mask contains no voxels on "
            "the right side of MNI x=0."
        )

    return left_mask, right_mask


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
):
    """
    Calculate LC contrast per axial slice and per hemisphere.contrast =(LC_slice_mean - DPT_slice_mean)/ DPT_slice_mean

    """

    subject = str(subject)
    session = str(session)

    tse_in_mni_path = Path(
        tse_in_mni_path
    )

    lc_mask_grid_path = Path(
        lc_mask_grid_path
    )

    dpt_mask_grid_path = Path(
        dpt_mask_grid_path
    )

    output_dir = Path(
        output_dir
    )

    required_inputs = {
        "TSE in MNI": tse_in_mni_path,
        "LC mask": lc_mask_grid_path,
        "DPT mask": dpt_mask_grid_path,
    }

    for name, path in required_inputs.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} does not exist: {path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_slice_path = (
        output_dir
        / "lc_contrast_per_slice.csv"
    )

    summary_path = (
        output_dir
        / "lc_peak_contrast_summary.json"
    )

    if (
        per_slice_path.exists()
        and summary_path.exists()
        and not overwrite
    ):
        with summary_path.open("r") as file:
            summary = json.load(file)

        per_slice_df = pd.read_csv(
            per_slice_path
        )

        print(
            f"\nRusing existing Step 8 results: "
            f"{subject}/{session}"
        )

        if show_per_slice:
            display_per_slice_table(
                per_slice_df=per_slice_df,
                subject=subject,
                session=session,
            )

        return summary, per_slice_df

    tse_image = ants.image_read(
        str(tse_in_mni_path)
    ).clone("float")

    lc_image = ants.image_read(
        str(lc_mask_grid_path)
    )

    dpt_image = ants.image_read(
        str(dpt_mask_grid_path)
    )

    if not same_grid(
        lc_image,
        tse_image,
    ):
        raise RuntimeError(
            "LC mask does not share the TSE grid."
        )

    if not same_grid(
        dpt_image,
        tse_image,
    ):
        raise RuntimeError(
            "DPT mask does not share the TSE grid."
        )

    tse = (
        tse_image.numpy()
        .astype(np.float64)
    )

    dpt_mask = (
        dpt_image.numpy() > 0
    )

    left_lc_mask, right_lc_mask = (
        split_bilateral_mask_by_mni_x(
            lc_image
        )
    )

    valid_tse = np.isfinite(tse)

    if exclude_zero_tse:
        valid_tse &= ~np.isclose(
            tse,
            0.0,
        )

    slice_rows = []

    for z_index in range(
        tse.shape[2]
    ):
        tse_slice = (
            tse[:, :, z_index]
        )

        valid_slice = (
            valid_tse[:, :, z_index]
        )

        dpt_slice = (
            dpt_mask[:, :, z_index]
            & valid_slice
        )

        left_slice = (
            left_lc_mask[:, :, z_index]
            & valid_slice
        )

        right_slice = (
            right_lc_mask[:, :, z_index]
            & valid_slice
        )

        dpt_voxels = int(
            dpt_slice.sum()
        )

        left_voxels = int(
            left_slice.sum()
        )

        right_voxels = int(
            right_slice.sum()
        )

        z_mm = (
            tse_image.origin[2]
            + tse_image.direction[2, 2]
            * z_index
            * tse_image.spacing[2]
        )

        dpt_mean = np.nan
        left_mean = np.nan
        right_mean = np.nan

        left_contrast = np.nan
        right_contrast = np.nan
        mean_lr_contrast = np.nan

        if dpt_voxels > 0:
            dpt_mean = float(
                tse_slice[
                    dpt_slice
                ].mean()
            )

        dpt_is_valid = (
            np.isfinite(dpt_mean)
            and not np.isclose(
                dpt_mean,
                0.0,
            )
        )

        if (
            left_voxels > 0
            and dpt_is_valid
        ):
            left_mean = float(
                tse_slice[
                    left_slice
                ].mean()
            )

            left_contrast = float(
                (
                    left_mean
                    - dpt_mean
                )
                / dpt_mean
            )

        if (
            right_voxels > 0
            and dpt_is_valid
        ):
            right_mean = float(
                tse_slice[
                    right_slice
                ].mean()
            )

            right_contrast = float(
                (
                    right_mean
                    - dpt_mean
                )
                / dpt_mean
            )

        left_valid = np.isfinite(
            left_contrast
        )

        right_valid = np.isfinite(
            right_contrast
        )

        if left_valid and right_valid:
            mean_lr_contrast = float(
                np.mean(
                    [
                        left_contrast,
                        right_contrast,
                    ]
                )
            )

        elif not require_both_sides:
            available = [
                value
                for value in (
                    left_contrast,
                    right_contrast,
                )
                if np.isfinite(value)
            ]

            if available:
                mean_lr_contrast = float(
                    np.mean(available)
                )

        slice_rows.append(
            {
                "subject": subject,
                "session": session,

                "z_index": int(
                    z_index
                ),
                "z_mni_mm": float(
                    z_mm
                ),

                "dpt_mean": dpt_mean,
                "dpt_voxels": dpt_voxels,

                "left_lc_mean": left_mean,
                "left_lc_voxels": left_voxels,
                "left_contrast": left_contrast,

                "right_lc_mean": right_mean,
                "right_lc_voxels": right_voxels,
                "right_contrast": right_contrast,

                "mean_lr_contrast": (
                    mean_lr_contrast
                ),
            }
        )

    per_slice_df = pd.DataFrame(
        slice_rows
    )

    valid_left = (
        per_slice_df
        .dropna(
            subset=["left_contrast"]
        )
    )

    valid_right = (
        per_slice_df
        .dropna(
            subset=["right_contrast"]
        )
    )

    valid_mean_lr = (
        per_slice_df
        .dropna(
            subset=["mean_lr_contrast"]
        )
    )

    if valid_left.empty:
        raise RuntimeError(
            "No valid left LC contrast values were calculated."
        )

    if valid_right.empty:
        raise RuntimeError(
            "No valid right LC contrast values were calculated."
        )

    if valid_mean_lr.empty:
        raise RuntimeError(
            "No valid bilateral mean LC contrast values "
            "were calculated."
        )

    left_peak_row = valid_left.loc[
        valid_left[
            "left_contrast"
        ].idxmax()
    ]

    right_peak_row = valid_right.loc[
        valid_right[
            "right_contrast"
        ].idxmax()
    ]

    literature_peak_row = valid_mean_lr.loc[
        valid_mean_lr[
            "mean_lr_contrast"
        ].idxmax()
    ]

    summary = {
        "literature_style_peak_mean_lr_contrast": float(
            literature_peak_row[
                "mean_lr_contrast"
            ]
        ),
        "literature_style_peak_z": int(
            literature_peak_row[
                "z_index"
            ]
        ),

        "left_peak_contrast": float(
            left_peak_row[
                "left_contrast"
            ]
        ),
        "left_peak_z": int(
            left_peak_row[
                "z_index"
            ]
        ),

        "right_peak_contrast": float(
            right_peak_row[
                "right_contrast"
            ]
        ),
        "right_peak_z": int(
            right_peak_row[
                "z_index"
            ]
        ),

        "subject": subject,
        "session": session,

        "literature_style_peak_z_mni_mm": float(
            literature_peak_row[
                "z_mni_mm"
            ]
        ),
        "left_peak_z_mni_mm": float(
            left_peak_row[
                "z_mni_mm"
            ]
        ),
        "right_peak_z_mni_mm": float(
            right_peak_row[
                "z_mni_mm"
            ]
        ),

        "number_of_valid_bilateral_slices": int(
            len(valid_mean_lr)
        ),

        "contrast_formula": (
            "(LC_slice_mean - DPT_slice_mean) "
            "/ DPT_slice_mean"
        ),
    }

    per_slice_df.to_csv(
        per_slice_path,
        index=False,
    )

    with summary_path.open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    if show_per_slice:
        display_per_slice_table(
            per_slice_df=per_slice_df,
            subject=subject,
            session=session,
        )

    summary_table = pd.DataFrame(
        [summary]
    )[
        [
            "subject",
            "session",
            "literature_style_peak_mean_lr_contrast",
            "literature_style_peak_z",
            "left_peak_contrast",
            "left_peak_z",
            "right_peak_contrast",
            "right_peak_z",
        ]
    ]

    print(
        f"\nPeak LC contrast summary: "
        f"{subject}/{session}"
    )

    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(summary_table)
        else:
            print(
                summary_table.to_string(
                    index=False
                )
            )

    except (ImportError, NameError):
        print(
            summary_table.to_string(
                index=False
            )
        )

    print(
        f"\nSaved per-slice results: "
        f"{per_slice_path}"
    )
    print(
        f"Saved peak summary: "
        f"{summary_path}"
    )

    return summary, per_slice_df