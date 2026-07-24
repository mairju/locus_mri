from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import config


def get_phase_from_session(session: str) -> str:
    if session not in config.SESSIONS:
        raise ValueError(f"Unknown session: {session}")
    return config.SESSIONS[session]


def get_nifti_paths(root: Path, session: str, modality: str, subject: str | None = None) -> list[str]:
    phase = get_phase_from_session(session)
    root = Path(root)
    search_root = root / subject if subject else root

    files = sorted(search_root.rglob(f"*{modality}*.nii.gz"))
    files = [str(f) for f in files if phase in str(f) and session in str(f)]
    return files


def build_session_dataframe(
    session=None,
    subject=None,
    modalities=None,
    save: bool = True,
    root: Path | None = None,
    output_root: Path | None = None,
) -> pd.DataFrame:
    modalities = modalities if modalities is not None else set(config.DEFAULT_KEYWORDS)
    root = root if root is not None else config.ROOT_NIFTI
    output_root = output_root if output_root is not None else config.OUT_ROOT

    if subject is None:
        subjects = [f"sub{i:03d}" for i in range(1, 85)]
    elif isinstance(subject, str):
        subjects = [subject]
    else:
        subjects = subject

    if session is None:
        sessions = list(config.SESSIONS.keys())
    elif isinstance(session, str):
        sessions = [session]
    else:
        sessions = session

    rows = []
    for ses in sessions:
        phase = get_phase_from_session(ses)
        for subj in subjects:
            row = {"subject": subj, "session": ses, "phase": phase}
            for modality in modalities:
                files = get_nifti_paths(root=root, session=ses, modality=modality, subject=subj)
                key = modality.lower()
                row[f"{key}_path"] = files[0] if files else False
            rows.append(row)

    df = pd.DataFrame(rows)

    if save:
        output_root.mkdir(parents=True, exist_ok=True)
        if session is None:
            filename = "all_sessions_input_files.csv"
        elif isinstance(session, str):
            filename = f"{session}_input_files.csv"
        else:
            filename = "selected_sessions_input_files.csv"
        df.to_csv(output_root / filename, index=False)

    return df
