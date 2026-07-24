from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import ants
import numpy as np

from . import config


def create_mni_brainstem_hires_grid(
    mni_template_path,
    brainstem_mask_mni_path,
    output_dir,
    target_spacing=config.TARGET_SPACING,
    dilation_iterations: int = config.BRAINSTEM_DILATION_ITERS_FSL,
    crop_padding_vox: int = config.CROP_PADDING_VOX,
    overwrite: bool = False,
) -> dict:

    mni_template_path = Path(
        mni_template_path
    ).expanduser().resolve()

    brainstem_mask_mni_path = Path(
        brainstem_mask_mni_path
    ).expanduser().resolve()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()

    if not mni_template_path.exists():
        raise FileNotFoundError(
            f"MNI template not found: {mni_template_path}"
        )

    if not brainstem_mask_mni_path.exists():
        raise FileNotFoundError(
            f"MNI brainstem mask not found: {brainstem_mask_mni_path}"
        )

    if dilation_iterations < 0:
        raise ValueError(
            "dilation_iterations must be >= 0."
        )

    if crop_padding_vox < 0:
        raise ValueError(
            "crop_padding_vox must be >= 0."
        )

    fsl_dir = Path(
        config.FSL_DIR
    ).expanduser().resolve()

    if not fsl_dir.exists():
        raise FileNotFoundError(
            f"Configured FSL directory does not exist: {fsl_dir}"
        )

    fsl_wrapper_dir = (
        fsl_dir
        / "share"
        / "fsl"
        / "bin"
    )

    fsl_bin_dir = (
        fsl_dir
        / "bin"
    )

    fslmaths = (
        fsl_wrapper_dir
        / "fslmaths"
    )

    if not fslmaths.exists():
        raise FileNotFoundError(
            f"'fslmaths' was not found at: {fslmaths}"
        )

    # Environment passed specifically to FSL subprocesses.
    fsl_env = os.environ.copy()
    fsl_env["FSLDIR"] = str(fsl_dir)
    fsl_env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    current_path = fsl_env.get("PATH", "")

    fsl_env["PATH"] = os.pathsep.join(
        [
            str(fsl_wrapper_dir),
            str(fsl_bin_dir),
            current_path,
        ]
    )

    extract_region = shutil.which(
        "ExtractRegionFromImageByMask"
    )

    if extract_region is None:
        raise RuntimeError(
            "'ExtractRegionFromImageByMask' was not found on PATH. "
            "Configure ANTs before running Step 5."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_dilated_path = (
        output_dir
        / "brainstem_mask_MNI_dilated.nii.gz"
    )

    mask_target_grid_path = (
        output_dir
        / "brainstem_mask_MNI_dilated_on_0p5mm_grid.nii.gz"
    )

    hires_grid_path = (
        output_dir
        / "MNI_brainstem_hires_grid_0p5mm.nii.gz"
    )

    manifest_path = (
        output_dir
        / "step5_hires_grid.json"
    )

    required_outputs = (
        mask_dilated_path,
        mask_target_grid_path,
        hires_grid_path,
        manifest_path,
    )

    if (
        not overwrite
        and all(path.exists() for path in required_outputs)
    ):
        with manifest_path.open("r") as file:
            return json.load(file)

    mni_image = ants.image_read(
        str(mni_template_path)
    ).clone("float")

    if mni_image.dimension != 3:
        raise ValueError(
            "Expected a 3D MNI template, found "
            f"{mni_image.dimension} dimensions."
        )

    if not np.allclose(
        np.asarray(mni_image.spacing),
        np.asarray(target_spacing),
        atol=1e-5,
    ):
        raise ValueError(
            "The supplied MNI template must already have "
            f"{target_spacing} mm isotropic spacing.\n"
            f"Observed spacing: {mni_image.spacing}\n"
            f"Template: {mni_template_path}"
        )

    print("FSL directory:       ", fsl_dir)
    print("fslmaths executable: ", fslmaths)
    print("MNI template spacing:", mni_image.spacing)
    print("MNI template shape:  ", mni_image.shape)

    dilation_options = [
        "-dilM"
    ] * dilation_iterations

    dilation_command = [
        str(fslmaths),
        str(brainstem_mask_mni_path),
        "-bin",
        *dilation_options,
        "-bin",
        str(mask_dilated_path),
    ]

    print(
        "Dilating MNI brainstem mask "
        f"{dilation_iterations} time(s)..."
    )

    dilation_result = subprocess.run(
        dilation_command,
        check=False,
        env=fsl_env,
        capture_output=True,
        text=True,
    )

    if dilation_result.returncode != 0:
        raise RuntimeError(
            "fslmaths dilation failed.\n\n"
            f"Command:\n{' '.join(dilation_command)}\n\n"
            f"stdout:\n{dilation_result.stdout}\n\n"
            f"stderr:\n{dilation_result.stderr}"
        )

    if not mask_dilated_path.exists():
        raise RuntimeError(
            "FSL dilation completed but did not produce: "
            f"{mask_dilated_path}"
        )

    brainstem_mask_dilated = ants.image_read(
        str(mask_dilated_path)
    ).clone("float")

    brainstem_mask_target = ants.resample_image_to_target(
        image=brainstem_mask_dilated,
        target=mni_image,
        interp_type="genericLabel",
    )

    brainstem_mask_target = ants.threshold_image(
        brainstem_mask_target,
        low_thresh=0.5,
        high_thresh=1e9,
        inval=1,
        outval=0,
    )

    mask_voxel_count = int(
        np.count_nonzero(
            brainstem_mask_target.numpy()
        )
    )

    if mask_voxel_count == 0:
        raise RuntimeError(
            "The brainstem mask became empty after resampling "
            "onto the 0.5 mm MNI grid."
        )

    ants.image_write(
        brainstem_mask_target,
        str(mask_target_grid_path),
    )

    crop_command = [
        extract_region,
        "3",
        str(mni_template_path),
        str(hires_grid_path),
        str(mask_target_grid_path),
        "1",
        str(crop_padding_vox),
    ]

    print(
        "Creating cropped 0.5 mm MNI reference grid..."
    )

    crop_result = subprocess.run(
        crop_command,
        check=False,
        capture_output=True,
        text=True,
    )

    if crop_result.returncode != 0:
        raise RuntimeError(
            "ExtractRegionFromImageByMask failed.\n\n"
            f"Command:\n{' '.join(crop_command)}\n\n"
            f"stdout:\n{crop_result.stdout}\n\n"
            f"stderr:\n{crop_result.stderr}"
        )

    if not hires_grid_path.exists():
        raise RuntimeError(
            "Cropping completed but did not produce: "
            f"{hires_grid_path}"
        )

    hires_grid = ants.image_read(
        str(hires_grid_path)
    )

    if not np.allclose(
        np.asarray(hires_grid.spacing),
        np.asarray(target_spacing),
        atol=1e-5,
    ):
        raise RuntimeError(
            "The cropped reference has unexpected spacing.\n"
            f"Expected: {target_spacing}\n"
            f"Observed: {hires_grid.spacing}"
        )

    if any(
        cropped_size > full_size
        for cropped_size, full_size in zip(
            hires_grid.shape,
            mni_image.shape,
        )
    ):
        raise RuntimeError(
            "The cropped reference is unexpectedly larger "
            "than the full MNI template."
        )

    info = {
        "source_mni_template": str(
            mni_template_path
        ),
        "brainstem_mask_mni": str(
            brainstem_mask_mni_path
        ),
        "brainstem_mask_mni_dilated": str(
            mask_dilated_path
        ),
        "brainstem_mask_on_target_grid": str(
            mask_target_grid_path
        ),
        "brainstem_hires_grid": str(
            hires_grid_path
        ),
        "fsl_dir": str(
            fsl_dir
        ),
        "fslmaths_executable": str(
            fslmaths
        ),
        "target_spacing": list(
            target_spacing
        ),
        "dilation_method": "FSL_dilM",
        "dilation_iterations": (
            dilation_iterations
        ),
        "crop_padding_vox": (
            crop_padding_vox
        ),
        "crop_padding_mm": [
            crop_padding_vox * spacing
            for spacing in target_spacing
        ],
        "mask_voxel_count_on_target_grid": (
            mask_voxel_count
        ),
        "full_reference_shape": list(
            mni_image.shape
        ),
        "cropped_grid_shape": list(
            hires_grid.shape
        ),
        "cropped_grid_spacing": list(
            hires_grid.spacing
        ),
        "cropped_grid_origin": list(
            hires_grid.origin
        ),
        "cropped_grid_direction": np.asarray(
            hires_grid.direction
        ).tolist(),
        "commands": {
            "dilation": dilation_command,
            "crop": crop_command,
        },
    }

    with manifest_path.open("w") as file:
        json.dump(
            info,
            file,
            indent=2,
        )

    print("Full MNI shape:    ", mni_image.shape)
    print("Cropped grid shape:", hires_grid.shape)
    print("Grid spacing:      ", hires_grid.spacing)
    print(
        "Crop padding:      ",
        crop_padding_vox,
        "voxels =",
        crop_padding_vox * target_spacing[0],
        "mm",
    )
    print("Saved grid:        ", hires_grid_path)

    return info