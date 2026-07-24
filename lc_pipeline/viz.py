from __future__ import annotations

import subprocess
from pathlib import Path

from . import config


def view_raw_inputs(row, mni_template: Path = config.MNI_TEMPLATE):
    """Step 0 QC: raw T1w + TSE overlaid on MNI, before anything is registered."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--robustRange",
        str(mni_template), "--name", "MNI", "--cmap", "greyscale", "--alpha", "100",
        str(row["t1w_path"]), "--name", "T1w", "--cmap", "red-yellow", "--alpha", "40", "--interpolation", "linear",
        str(row["tse_path"]), "--name", "TSE", "--cmap", "blue-lightblue", "--alpha", "55", "--interpolation", "linear",
    ])


def view_t1_in_mni(info_t1_mni: dict, mni_template: Path = config.MNI_TEMPLATE):
    """Step 2 QC: warped T1w overlaid on the MNI template."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--robustRange",
        str(mni_template), "--name", "MNI", "--cmap", "greyscale", "--alpha", "100",
        str(info_t1_mni["warpedmovout"]), "--name", "T1w in MNI", "--cmap", "greyscale", "--alpha", "50", "--interpolation", "linear",
    ])


def view_brainstem_mask_mni(brainstem_mask_path: Path, mni_template: Path = config.MNI_TEMPLATE):
    """Brainstem-mask-creation QC: mask overlaid on the MNI template."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(mni_template),
        str(mni_template), "--name", "MNI template", "--cmap", "greyscale", "--alpha", "100",
        str(brainstem_mask_path), "--name", "Brainstem mask", "--overlayType", "mask",
        "--maskColour", "1", "0", "0", "--threshold", "0.5", "1.5", "--alpha", "45", "--interpolation", "none",
    ])


def view_brainstem_in_t1w(t1w_n4_path: Path, info_brainstem: dict, session_label: str = ""):
    """Step 3 QC: undilated + dilated brainstem mask in T1w space."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(t1w_n4_path), "--robustRange",
        str(t1w_n4_path), "--name", f"{session_label}_T1w_N4", "--cmap", "greyscale", "--alpha", "100",
        info_brainstem["brainstem_mask_t1w"], "--name", "Brainstem in T1w", "--overlayType", "mask",
        "--maskColour", "1", "0", "0", "--threshold", "0.5", "1.5", "--alpha", "100",
        "--outline", "--outlineWidth", "2", "--interpolation", "none",
        info_brainstem["brainstem_mask_t1w_dilated"], "--name", "Brainstem dilated", "--overlayType", "mask",
        "--maskColour", "0", "0", "1", "--threshold", "0.5", "1.5", "--alpha", "30", "--interpolation", "none",
    ])


def view_tse_in_t1w(t1w_n4_path: Path, info_tse_t1w: dict, info_brainstem: dict, label: str = ""):
    """Step 4 QC: TSE registered into T1w space, with the registration constraint mask."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(t1w_n4_path), "--robustRange",
        str(t1w_n4_path), "--name", f"{label}_T1w_N4", "--cmap", "greyscale", "--alpha", "100",
        info_tse_t1w["warped_tse"], "--name", "TSE_in_T1w", "--cmap", "blue-lightblue", "--alpha", "50", "--interpolation", "linear",
        info_brainstem["brainstem_mask_t1w_dilated"], "--name", "Dilated brainstem constraint", "--overlayType", "mask",
        "--maskColour", "1", "0", "0", "--threshold", "0.5", "1.5", "--alpha", "100",
    ])


def view_hires_grid(info_hires_grid: dict, mni_template: Path = config.MNI_TEMPLATE):
    """Step 5 QC: full MNI template, the dilated crop mask, and the final cropped grid."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(mni_template), "--robustRange",
        str(mni_template), "--name", "Full MNI 0.5mm", "--cmap", "greyscale", "--alpha", "100",
        str(info_hires_grid["brainstem_mask_on_target_grid"]), "--name", "Dilated brainstem crop mask",
        "--overlayType", "mask", "--maskColour", "1", "0", "0", "--threshold", "0.5", "1.5", "--alpha", "100",
        "--outline", "--outlineWidth", "2",
        str(info_hires_grid["brainstem_hires_grid"]), "--name", "Cropped MNI brainstem grid",
        "--cmap", "blue-lightblue", "--alpha", "35", "--interpolation", "linear",
    ])


def view_tse_in_mni(info_tse_mni: dict, brainstem_mask_mni_path: Path, mni_template: Path = config.MNI_TEMPLATE):
    """Step 6 QC: composed-resampled TSE in MNI space, with the brainstem mask for reference."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(mni_template), "--robustRange",
        str(mni_template), "--cmap", "greyscale", "--alpha", "100",
        str(info_tse_mni["output_tse_mni"]), "--name", "TSE in MNI", "--cmap", "blue-lightblue", "--alpha", "45", "--interpolation", "linear",
        str(brainstem_mask_mni_path), "--name", "Brainstem mask", "--overlayType", "mask",
        "--maskColour", "1", "0", "0", "--threshold", "0.5", "1.5", "--alpha", "100",
        "--outline", "--outlineWidth", "2", "--interpolation", "none",
    ])


def view_masks_on_tse_grid(info_tse_mni: dict, info_masks_grid: dict, label: str = ""):
    """Step 7 QC: final TSE with LC and DPT masks on the same grid."""
    return subprocess.Popen([
        "fsleyes", "--scene", "ortho", "--layout", "horizontal", "--displaySpace", str(info_tse_mni["output_tse_mni"]), "--robustRange",
        str(info_tse_mni["output_tse_mni"]), "--name", f"{label}_TSE_in_MNI", "--cmap", "greyscale", "--alpha", "100",
        str(info_masks_grid["lc_mask_grid"]), "--name", "LC mask", "--overlayType", "mask",
        "--maskColour", "1", "0", "0", "--threshold", "0.5", "2.5", "--alpha", "100",
        "--outline", "--outlineWidth", "2", "--interpolation", "none",
        str(info_masks_grid["dpt_mask_grid"]), "--name", "DPT mask", "--overlayType", "mask",
        "--maskColour", "0", "0", "1", "--threshold", "0.5", "2.5", "--alpha", "100",
        "--outline", "--outlineWidth", "2", "--interpolation", "none",
    ])
