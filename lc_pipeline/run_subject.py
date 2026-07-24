from __future__ import annotations

from pathlib import Path

from . import config
from .brainstem_mask_creation import create_brainstem_mask
from .step1_n4 import n4_bias_correction
from .step2_t1_to_mni import register_t1w_n4_to_mni
from .step3_brainstem_mask import create_brainstem_masks_in_t1w
from .step4_tse_to_t1 import register_tse_to_t1w
from .step5_hires_grid import create_mni_brainstem_hires_grid
from .step6_resample_tse import resample_native_tse_to_mni
from .step7_masks_to_grid import put_lc_and_dpt_masks_on_tse_grid
from .step8_extract_cr import extract_literature_style_peak_contrast


def run_session(
    subject: str,
    session: str,
    t1w_path,
    tse_path,
    out_root: Path = config.OUT_ROOT,
    mni_template: Path = config.MNI_TEMPLATE,
    lc_mask: Path = config.LC_MASK_BOTH,
    dpt_mask: Path = config.DPT_MASK,
    brainstem_mask_mni_path: Path = config.BRAINSTEM_MASK_MNI_PATH,
    step2_transform: str = config.STEP2_TRANSFORM,
    overwrite: bool = False,
) -> dict:

    out_root = Path(out_root)
    t1w_path = Path(t1w_path)
    tse_path = Path(tse_path)

    print(f"\n{'='*60}\n{subject} / {session}\n{'='*60}")

    brainstem_mask_mni_path = Path(brainstem_mask_mni_path)
    if not brainstem_mask_mni_path.exists():
        print("Brainstem mask not found -- creating via Harvard-Oxford/FSL...")
        create_brainstem_mask(brainstem_mask_mni_path, overwrite=False)

    t1w_n4_path = out_root / "step1_n4" / "t1w_n4" / subject / session / "t1w_n4.nii.gz"
    if not t1w_n4_path.exists() or overwrite:
        t1w_n4_path = n4_bias_correction(input_path=t1w_path, output_path=t1w_n4_path, save=True)

    step2_outdir = out_root / "step2_t1w_mni" / subject / session
    info_t1_mni = register_t1w_n4_to_mni(
        path_fixed_img=mni_template, path_moving_img=t1w_n4_path, outdir=step2_outdir,
        type_of_transform=step2_transform, force=overwrite,
    )

    step3_outdir = out_root / "step3_brainstem_t1w" / subject / session
    info_brainstem = create_brainstem_masks_in_t1w(
        brainstem_mask_mni_path=brainstem_mask_mni_path, t1w_n4_path=t1w_n4_path,
        inverse_transforms=info_t1_mni["invtransforms"], output_dir=step3_outdir, overwrite=overwrite,
    )

    step4_outdir = out_root / "step4_tse_t1w" / subject / session
    info_tse_t1w = register_tse_to_t1w(
        t1w_n4_path=t1w_n4_path, tse_path=tse_path,
        brainstem_mask_dilated_path=info_brainstem["brainstem_mask_t1w_dilated"],
        output_dir=step4_outdir, overwrite=overwrite,
    )

    step5_outdir = out_root / "step5_hires_grid"
    info_hires_grid = create_mni_brainstem_hires_grid(
        mni_template_path=mni_template, brainstem_mask_mni_path=brainstem_mask_mni_path,
        output_dir=step5_outdir, overwrite=False,
    )

    step6_outdir = out_root / "step6_tse_mni" / subject / session
    info_tse_mni = resample_native_tse_to_mni(
        tse_native_path=tse_path,
        mni_hires_grid_path=info_hires_grid["brainstem_hires_grid"],
        t1_to_mni_warp_path=info_t1_mni["warp01"],
        t1_to_mni_affine_path=info_t1_mni["affine01"],
        tse_to_t1_affine_path=info_tse_t1w["affine_transform"],
        output_dir=step6_outdir, overwrite=overwrite,
    )

    step7_outdir = out_root / "step7_masks_grid" / subject / session
    info_masks_grid = put_lc_and_dpt_masks_on_tse_grid(
        lc_mask_mni_path=lc_mask, dpt_mask_mni_path=dpt_mask,
        tse_in_mni_path=info_tse_mni["output_tse_mni"], output_dir=step7_outdir, overwrite=overwrite,
    )

    step8_outdir = out_root / "step8_contrast" / subject / session
    summary, per_slice_df = extract_literature_style_peak_contrast(
        subject=subject, session=session,
        tse_in_mni_path=info_tse_mni["output_tse_mni"],
        lc_mask_grid_path=info_masks_grid["lc_mask_grid"],
        dpt_mask_grid_path=info_masks_grid["dpt_mask_grid"],
        output_dir=step8_outdir, overwrite=overwrite,
    )

    return {
        "subject": subject, "session": session,
        "summary": summary, "per_slice_df": per_slice_df,
        "t1w_n4_path": str(t1w_n4_path),
        "info_t1_mni": info_t1_mni,
        "info_brainstem": info_brainstem,
        "info_tse_t1w": info_tse_t1w,
        "info_hires_grid": info_hires_grid,
        "info_tse_mni": info_tse_mni,
        "info_masks_grid": info_masks_grid,
    }
