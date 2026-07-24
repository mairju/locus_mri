from __future__ import annotations

from pathlib import Path

import ants
import pandas as pd
from tqdm.auto import tqdm

from . import config


def n4_bias_correction(
    input_path,
    output_path,
    save: bool,
    rescale_intensities: bool = True,
    shrink_factor: int = config.SHRINK_FACTOR,
    convergence: dict = config.CONVERGENCE,
    spline_param: int = config.SPLINE_PARAMETER,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = ants.image_read(str(input_path))

    img_n4 = ants.n4_bias_field_correction(
        image=img,
        shrink_factor=shrink_factor,
        convergence=convergence,
        spline_param=spline_param,
        rescale_intensities=rescale_intensities,
    )

    if save:
        ants.image_write(img_n4, str(output_path))

    return output_path


def run_n4_bias_correction(
    df: pd.DataFrame,
    output_root: Path = config.OUT_ROOT,
    subject=None,
    session=None,
    save: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    df_run = df.copy()

    if subject is not None:
        if isinstance(subject, str):
            subject = [subject]
        df_run = df_run[df_run["subject"].isin(subject)].copy()

    if session is not None:
        if isinstance(session, str):
            session = [session]
        df_run = df_run[df_run["session"].isin(session)].copy()

    results = []

    for _, row in tqdm(
        df_run.iterrows(), total=len(df_run),
        desc="Applying N4 bias-field correction on T1w images", unit="T1w",
    ):
        subject_i = row["subject"]
        session_i = row["session"]
        t1w_path = row["t1w_path"]

        row_result = row.to_dict()

        if t1w_path is False or t1w_path == "False" or pd.isna(t1w_path):
            row_result["t1w_n4_path"] = False
            results.append(row_result)
            continue

        t1w_path = Path(t1w_path)
        output_dir = Path(output_root) / "step1_n4" / "t1w_n4" / subject_i / session_i
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "t1w_n4.nii.gz"

        if output_path.exists() and not overwrite:
            pass
        else:
            output_path = n4_bias_correction(input_path=t1w_path, output_path=output_path, save=save)

        row_result["t1w_n4_path"] = str(output_path)
        results.append(row_result)

    return pd.DataFrame(results)
