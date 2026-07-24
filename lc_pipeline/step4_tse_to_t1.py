from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path


def register_tse_to_t1w(
    t1w_n4_path,
    tse_path,
    brainstem_mask_dilated_path,
    output_dir,
    overwrite: bool = False,
) -> dict:
    t1w_n4_path = Path(t1w_n4_path)
    tse_path = Path(tse_path)
    brainstem_mask_dilated_path = Path(brainstem_mask_dilated_path)
    output_dir = Path(output_dir)

    for name, path in {
        "T1w N4": t1w_n4_path, "TSE": tse_path, "Dilated brainstem mask": brainstem_mask_dilated_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    ants_registration = shutil.which("antsRegistration")
    if ants_registration is None:
        raise RuntimeError("'antsRegistration' was not found on PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)
    transform_prefix = output_dir / "transform02_"
    affine_path = output_dir / "transform02_0GenericAffine.mat"
    warped_tse_path = output_dir / "tse2t1.nii.gz"
    manifest_path = output_dir / "TSE_to_T1w_registration.json"

    if affine_path.exists() and warped_tse_path.exists() and manifest_path.exists() and not overwrite:
        with manifest_path.open("r") as file:
            return json.load(file)

    fixed = str(t1w_n4_path)
    moving = str(tse_path)
    mask = str(brainstem_mask_dilated_path)

    command = [
        ants_registration,
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{transform_prefix},{warped_tse_path}]",
        "--interpolation", "LanczosWindowedSinc",
        "--use-histogram-matching", "0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--initial-moving-transform", f"[{fixed},{moving},1]",

        ## Rigid
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]", ## 0.50
        "--convergence", "[1000x500x250,1e-6,10]",
        "--shrink-factors", "4x2x1",
        "--smoothing-sigmas", "2x1x0vox",
        "--masks", f"[{mask},NULL]",

        ## Affine
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]", ## 0,50
        "--convergence", "[1000x500x250,1e-6,10]",
        "--shrink-factors", "4x2x1",
        "--smoothing-sigmas", "2x1x0vox",
        "--masks", f"[{mask},NULL]",
        "--verbose", "1",
    ]

    print("Running TSE -> T1w registration:\n")
    print(" ".join(shlex.quote(a) for a in command))
    print()

    subprocess.run(command, check=True)

    if not affine_path.exists():
        raise RuntimeError(f"Registration completed, but the affine transform was not created: {affine_path}")
    if not warped_tse_path.exists():
        raise RuntimeError(f"Registration completed, but the warped TSE was not created: {warped_tse_path}")

    info = {
        "fixed_t1w": str(t1w_n4_path),
        "moving_tse": str(tse_path),
        "fixed_mask": str(brainstem_mask_dilated_path),
        "forward_direction": "TSE_to_T1w",
        "transform_type": "rigid_plus_affine",
        "affine_transform": str(affine_path),
        "warped_tse": str(warped_tse_path),
        "forward_transforms": [str(affine_path)],
        "inverse_transforms": [str(affine_path)],
        "parameters": {
            "metric": "MI", "metric_bins": 32, "sampling_strategy": "Regular", "sampling_fraction": 0.25,
            "convergence": [1000, 500, 250], "convergence_threshold": 1e-6, "convergence_window": 10,
            "shrink_factors": [4, 2, 1], "smoothing_sigmas_vox": [2, 1, 0],
            "winsorization": [0.005, 0.995], "histogram_matching": False,
            "output_interpolation": "LanczosWindowedSinc",
        },
        "command": command,
    }

    with manifest_path.open("w") as file:
        json.dump(info, file, indent=2)

    print("\nSaved affine transform:", affine_path)
    print("Saved registered TSE:  ", warped_tse_path)
    print("Saved manifest:        ", manifest_path)
    return info
