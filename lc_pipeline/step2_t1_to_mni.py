from __future__ import annotations

import json
from pathlib import Path

import ants

from . import config


def register_t1w_n4_to_mni(
    path_fixed_img,
    path_moving_img,
    outdir,
    type_of_transform: str = config.STEP2_TRANSFORM,
    force: bool = False,
) -> dict:

    path_fixed_img = Path(path_fixed_img)
    path_moving_img = Path(path_moving_img)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    warped_path = outdir / "T1w_N4_in_MNI152_0p5mm.nii.gz"
    transform_json = outdir / "T1w_N4_to_MNI_transforms.json"

    if warped_path.exists() and transform_json.exists() and not force:
        print("Already registered:", warped_path)
        with open(transform_json, "r") as f:
            return json.load(f)

    if not path_fixed_img.exists():
        raise FileNotFoundError(f"Fixed image does not exist: {path_fixed_img}")
    if not path_moving_img.exists():
        raise FileNotFoundError(f"Moving image does not exist: {path_moving_img}")

    print("Fixed image : MNI152 template")
    print("Moving image:", path_moving_img)
    print("Transform   :", type_of_transform)

    fixed_img = ants.image_read(str(path_fixed_img)).clone("float")
    moving_img = ants.image_read(str(path_moving_img)).clone("float")

    reg = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform=type_of_transform,
        outprefix=str(outdir / "transform01_"),
        verbose=True,
    )

    ants.image_write(reg["warpedmovout"], str(warped_path))

    fwd = reg["fwdtransforms"]
    inv = reg["invtransforms"]

    warp01 = next(p for p in fwd if p.endswith("1Warp.nii.gz"))
    affine01 = next(p for p in fwd if p.endswith("GenericAffine.mat"))
    invwarp01 = next(p for p in inv if p.endswith("1InverseWarp.nii.gz"))

    info = {
        "fixed": str(path_fixed_img),
        "moving": str(path_moving_img),
        "type_of_transform": type_of_transform,
        "warpedmovout": str(warped_path),
        "fwdtransforms": fwd,
        "invtransforms": inv,
        "affine01": affine01,
        "warp01": warp01,
        "invwarp01": invwarp01,
    }

    with open(transform_json, "w") as f:
        json.dump(info, f, indent=2)

    print("Saved warped image:", warped_path)
    print("Saved transform info:", transform_json)
    return info
