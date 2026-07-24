from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ants
import numpy as np

from .utils import get_discrete_labels, same_grid


def put_label_mask_on_reference_grid(mask_path, reference_path, output_path, mask_name, overwrite: bool = False) -> dict:
    mask_path = Path(mask_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"{mask_name} not found: {mask_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_mask = ants.image_read(str(mask_path))
    reference = ants.image_read(str(reference_path))

    source_labels = get_discrete_labels(source_mask, mask_name)
    source_already_matches = same_grid(source_mask, reference)

    print(f"\n{mask_name}")
    print("  Source shape:   ", source_mask.shape)
    print("  Source spacing: ", source_mask.spacing)
    print("  Source origin:  ", source_mask.origin)
    print("  Labels:         ", source_labels)
    print("  Already matches:", source_already_matches)

    if output_path.exists() and not overwrite:
        existing_output = ants.image_read(str(output_path))
        if not same_grid(existing_output, reference):
            raise RuntimeError(
                f"Existing {mask_name} output does not match the reference grid:\n{output_path}\n"
                "Use overwrite=True to recreate it."
            )
        action = "reused_existing_output"
    elif source_already_matches:
        shutil.copy2(mask_path, output_path)
        action = "copied_without_resampling"
    else:
        ants_apply_transforms = shutil.which("antsApplyTransforms")
        if ants_apply_transforms is None:
            raise RuntimeError("'antsApplyTransforms' was not found on PATH.")
        command = [
            ants_apply_transforms, "-d", "3",
            "-i", str(mask_path), "-r", str(reference_path), "-o", str(output_path),
            "-n", "GenericLabel", "-t", "identity",
        ]
        subprocess.run(command, check=True)
        action = "identity_resampled"

    if not output_path.exists():
        raise RuntimeError(f"{mask_name} output was not created: {output_path}")

    output_mask = ants.image_read(str(output_path))
    if not same_grid(output_mask, reference):
        raise RuntimeError(f"{mask_name} output does not match the reference TSE grid.")

    output_labels = get_discrete_labels(output_mask, f"{mask_name} output")
    missing_labels = set(source_labels) - set(output_labels)
    if missing_labels:
        raise RuntimeError(
            f"{mask_name} lost these labels after grid matching: {sorted(missing_labels)}.\n"
            "The mask may have been clipped by the cropped TSE grid."
        )

    output_voxels = int(np.count_nonzero(output_mask.numpy()))

    print("  Action:        ", action)
    print("  Output shape:  ", output_mask.shape)
    print("  Output spacing:", output_mask.spacing)
    print("  Output origin: ", output_mask.origin)
    print("  Output labels: ", output_labels)
    print("  Output voxels: ", output_voxels)
    print("  Saved:         ", output_path)

    return {
        "source_path": str(mask_path), "output_path": str(output_path), "action": action,
        "source_already_matched": source_already_matches, "source_labels": source_labels,
        "output_labels": output_labels, "output_voxel_count": output_voxels,
    }


def put_lc_and_dpt_masks_on_tse_grid(lc_mask_mni_path, dpt_mask_mni_path, tse_in_mni_path, output_dir, overwrite: bool = False) -> dict:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lc_output_path = output_dir / "LC_mask_grid.nii.gz"
    dpt_output_path = output_dir / "DPT_mask_grid.nii.gz"

    lc_info = put_label_mask_on_reference_grid(
        mask_path=lc_mask_mni_path, reference_path=tse_in_mni_path, output_path=lc_output_path,
        mask_name="LC mask", overwrite=overwrite,
    )
    dpt_info = put_label_mask_on_reference_grid(
        mask_path=dpt_mask_mni_path, reference_path=tse_in_mni_path, output_path=dpt_output_path,
        mask_name="DPT mask", overwrite=overwrite,
    )

    reference = ants.image_read(str(tse_in_mni_path))

    info = {
        "reference_tse_mni": str(tse_in_mni_path),
        "lc_mask_source": str(lc_mask_mni_path),
        "lc_mask_grid": str(lc_output_path),
        "lc_action": lc_info["action"],
        "lc_labels": lc_info["output_labels"],
        "lc_voxel_count": lc_info["output_voxel_count"],
        "dpt_mask_source": str(dpt_mask_mni_path),
        "dpt_mask_grid": str(dpt_output_path),
        "dpt_action": dpt_info["action"],
        "dpt_labels": dpt_info["output_labels"],
        "dpt_voxel_count": dpt_info["output_voxel_count"],
        "interpolator": "GenericLabel",
        "transform": "identity",
        "target_shape": list(reference.shape),
        "target_spacing": list(reference.spacing),
        "target_origin": list(reference.origin),
        "target_direction": np.asarray(reference.direction).tolist(),
    }

    manifest_path = output_dir / "step7_masks_on_tse_grid.json"
    with manifest_path.open("w") as file:
        json.dump(info, file, indent=2)

    return info
