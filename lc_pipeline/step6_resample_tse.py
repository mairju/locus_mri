from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import ants
import numpy as np


def resample_native_tse_to_mni(
    tse_native_path,
    mni_hires_grid_path,
    t1_to_mni_warp_path,
    t1_to_mni_affine_path,
    tse_to_t1_affine_path,
    output_dir,
    overwrite: bool = False,
) -> dict:
    tse_native_path = Path(tse_native_path)
    mni_hires_grid_path = Path(mni_hires_grid_path)
    t1_to_mni_warp_path = Path(t1_to_mni_warp_path)
    t1_to_mni_affine_path = Path(t1_to_mni_affine_path)
    tse_to_t1_affine_path = Path(tse_to_t1_affine_path)
    output_dir = Path(output_dir)

    required_paths = {
        "Native TSE": tse_native_path, "MNI high-resolution grid": mni_hires_grid_path,
        "T1w-to-MNI warp": t1_to_mni_warp_path, "T1w-to-MNI affine": t1_to_mni_affine_path,
        "TSE-to-T1w affine": tse_to_t1_affine_path,
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    ants_apply_transforms = shutil.which("antsApplyTransforms")
    if ants_apply_transforms is None:
        raise RuntimeError("'antsApplyTransforms' was not found on PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tse_in_MNI_brainstem_0p5mm.nii.gz"
    manifest_path = output_dir / "step6_tse_to_mni.json"

    if output_path.exists() and manifest_path.exists() and not overwrite:
        with manifest_path.open("r") as file:
            return json.load(file)

    transforms = [t1_to_mni_warp_path, t1_to_mni_affine_path, tse_to_t1_affine_path]

    command = [
        ants_apply_transforms, "-d", "3",
        "-i", str(tse_native_path),          # always the ORIGINAL native TSE
        "-r", str(mni_hires_grid_path),
        "-o", str(output_path),
        "-n", "LanczosWindowedSinc",
        "-t", str(transforms[0]),
        "-t", str(transforms[1]),
        "-t", str(transforms[2]),
        "--default-value", "0",
        "--verbose", "1",
    ]

    print("Applying one composed transform chain:")
    print("  TSE native -> T1w -> MNI")
    print()
    print("Input TSE:       ", tse_native_path)
    print("Reference grid:  ", mni_hires_grid_path)
    print("TSE -> T1 affine:", tse_to_t1_affine_path)
    print("T1 -> MNI affine:", t1_to_mni_affine_path)
    print("T1 -> MNI warp:  ", t1_to_mni_warp_path)

    start_time = perf_counter()
    subprocess.run(command, check=True)
    runtime_seconds = perf_counter() - start_time

    if not output_path.exists():
        raise RuntimeError(f"antsApplyTransforms completed but did not create: {output_path}")

    reference_image = ants.image_read(str(mni_hires_grid_path))
    output_image = ants.image_read(str(output_path))

    geometry_checks = {
        "same_shape": tuple(output_image.shape) == tuple(reference_image.shape),
        "same_spacing": np.allclose(output_image.spacing, reference_image.spacing, atol=1e-5),
        "same_origin": np.allclose(output_image.origin, reference_image.origin, atol=1e-5),
        "same_direction": np.allclose(np.asarray(output_image.direction), np.asarray(reference_image.direction), atol=1e-5),
    }
    if not all(geometry_checks.values()):
        raise RuntimeError(f"The transformed TSE does not match the reference grid geometry:\n{geometry_checks}")

    output_array = output_image.numpy()
    if not np.isfinite(output_array).all():
        raise RuntimeError("The transformed TSE contains NaN or infinite values.")
    if np.count_nonzero(output_array) == 0:
        raise RuntimeError("The transformed TSE is completely empty.")

    info = {
        "input_tse_native": str(tse_native_path),
        "reference_grid": str(mni_hires_grid_path),
        "output_tse_mni": str(output_path),
        "transform_direction": "TSE_to_T1w_to_MNI",
        "transformlist": [str(p) for p in transforms],
        "tse_to_t1_affine": str(tse_to_t1_affine_path),
        "t1_to_mni_affine": str(t1_to_mni_affine_path),
        "t1_to_mni_warp": str(t1_to_mni_warp_path),
        "interpolator": "LanczosWindowedSinc",
        "number_of_resampling_operations": 1,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60,
        "output_shape": list(output_image.shape),
        "output_spacing": list(output_image.spacing),
        "output_origin": list(output_image.origin),
        "output_direction": np.asarray(output_image.direction).tolist(),
        "geometry_checks": geometry_checks,
        "command": command,
    }

    with manifest_path.open("w") as file:
        json.dump(info, file, indent=2)

    print()
    print("Saved TSE in MNI:", output_path)
    print(f"Runtime: {runtime_seconds / 60:.2f} minutes")
    print("Output shape:  ", output_image.shape)
    print("Output spacing:", output_image.spacing)
    return info
