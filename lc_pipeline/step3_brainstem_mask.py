from __future__ import annotations

from pathlib import Path

import ants

from . import config


def create_brainstem_masks_in_t1w(
    brainstem_mask_mni_path,
    t1w_n4_path,
    inverse_transforms,
    output_dir,
    dilation_radius: int = config.BRAINSTEM_DILATION_RADIUS_VOX,
    overwrite: bool = False,
) -> dict:
    brainstem_mask_mni_path = Path(brainstem_mask_mni_path)
    t1w_n4_path = Path(t1w_n4_path)
    output_dir = Path(output_dir)

    if not brainstem_mask_mni_path.exists():
        raise FileNotFoundError(f"MNI brainstem mask not found: {brainstem_mask_mni_path}")
    if not t1w_n4_path.exists():
        raise FileNotFoundError(f"T1w N4 image not found: {t1w_n4_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_t1w_path = output_dir / "bstem_in_t1.nii.gz"
    mask_t1w_dilated_path = output_dir / "bstem_dil.nii.gz"

    if mask_t1w_path.exists() and mask_t1w_dilated_path.exists() and not overwrite:
        print("Brainstem masks already exist.")
        return {
            "brainstem_mask_t1w": str(mask_t1w_path),
            "brainstem_mask_t1w_dilated": str(mask_t1w_dilated_path),
        }

    inverse_transforms = [str(Path(p)) for p in inverse_transforms]
    missing = [p for p in inverse_transforms if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing inverse transforms:\n" + "\n".join(missing))

    affine_matches = [p for p in inverse_transforms if p.endswith("GenericAffine.mat")]
    inverse_warp_matches = [p for p in inverse_transforms if p.endswith("InverseWarp.nii.gz")]

    if len(affine_matches) != 1:
        raise RuntimeError(f"Expected exactly one affine transform, found: {affine_matches}")
    if len(inverse_warp_matches) != 1:
        raise RuntimeError(f"Expected exactly one inverse warp, found: {inverse_warp_matches}")

    affine_path = affine_matches[0]
    inverse_warp_path = inverse_warp_matches[0]

    print("Reference grid:", t1w_n4_path)
    print("Input mask:    ", brainstem_mask_mni_path)
    print("Affine:       ", affine_path)
    print("Inverse warp: ", inverse_warp_path)

    t1w_n4 = ants.image_read(str(t1w_n4_path)).clone("float")
    brainstem_mni = ants.image_read(str(brainstem_mask_mni_path)).clone("float")

    brainstem_t1w = ants.apply_transforms(
        fixed=t1w_n4,
        moving=brainstem_mni,
        transformlist=[affine_path, inverse_warp_path],
        whichtoinvert=[True, False],
        interpolator="genericLabel",
        defaultvalue=0,
        verbose=False,
    )
    brainstem_t1w = ants.threshold_image(brainstem_t1w, low_thresh=0.5, high_thresh=1.5, inval=1, outval=0)
    ants.image_write(brainstem_t1w, str(mask_t1w_path))

    brainstem_t1w_dilated = ants.morphology(
        brainstem_t1w, operation="dilate", radius=dilation_radius, mtype="binary", value=1, shape="ball",
    )
    brainstem_t1w_dilated = ants.threshold_image(
        brainstem_t1w_dilated, low_thresh=0.5, high_thresh=1.5, inval=1, outval=0,
    )
    ants.image_write(brainstem_t1w_dilated, str(mask_t1w_dilated_path))

    print("Saved T1w-space mask:", mask_t1w_path)
    print("Saved dilated mask:  ", mask_t1w_dilated_path)

    return {
        "brainstem_mask_mni": str(brainstem_mask_mni_path),
        "brainstem_mask_t1w": str(mask_t1w_path),
        "brainstem_mask_t1w_dilated": str(mask_t1w_dilated_path),
        "dilation_radius": dilation_radius,
        "inverse_transforms": [affine_path, inverse_warp_path],
    }
