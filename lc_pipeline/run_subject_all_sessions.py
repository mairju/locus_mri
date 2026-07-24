#!/usr/bin/env python3
"""

python3 -m lc_pipeline.run_subject_all_sessions --subject sub002
python3 -m lc_pipeline.run_subject_all_sessions --subject sub002 --sessions ses001 ses002
python3 -m lc_pipeline.run_subject_all_sessions --subject sub002 --overwrite
python3 -m lc_pipeline.run_subject_all_sessions --subject sub002 --sessions ses001 --out-root /home/maria/Documents/data/hiwi_sample/24.07_ses001 --full-syn
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from . import config
from .data_discovery import build_session_dataframe
from .run_subject import run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", required=True, help="e.g. sub002")
    parser.add_argument("--sessions", nargs="*", default=None,
                         help="Session IDs to run (default: all sessions in config.SESSIONS)")
    parser.add_argument("--root-nifti", default=str(config.ROOT_NIFTI), help="Root of raw NIfTI tree")
    parser.add_argument("--out-root", default=str(config.OUT_ROOT), help="Output/derivatives root")
    parser.add_argument("--overwrite", action="store_true", help="Recompute steps even if outputs already exist")
    parser.add_argument("--full-syn", action="store_true",
                         help="Use antsRegistrationSyN[s] (full, slow) instead of the Quick preset for Step 2")
    args = parser.parse_args()

    subject = args.subject
    sessions = args.sessions if args.sessions else list(config.SESSIONS.keys())
    out_root = Path(args.out_root)
    step2_transform = config.STEP2_TRANSFORM_FULL if args.full_syn else config.STEP2_TRANSFORM_QUICK

    print(f"Subject:  {subject}")
    print(f"Sessions: {sessions}")
    print(f"Out root: {out_root}")
    print(f"Step 2 transform: {step2_transform}")

    df = build_session_dataframe(
        session=sessions, subject=subject, root=Path(args.root_nifti), output_root=out_root, save=True,
    )

    session_summaries = []
    failures = []

    for session in sessions:
        row_matches = df[(df["subject"] == subject) & (df["session"] == session)]
        if row_matches.empty:
            print(f"[{subject}/{session}] no dataframe row found -- skipping")
            failures.append({"subject": subject, "session": session, "error": "no matching row in discovered dataframe"})
            continue

        row = row_matches.iloc[0]
        t1w_path = row.get("t1w_path")
        tse_path = row.get("tse_path")

        if not t1w_path or t1w_path is False or not tse_path or tse_path is False:
            print(f"[{subject}/{session}] missing T1w or TSE file -- skipping")
            failures.append({"subject": subject, "session": session, "error": "missing t1w_path or tse_path"})
            continue

        try:
            result = run_session(
                subject=subject, session=session, t1w_path=t1w_path, tse_path=tse_path,
                out_root=out_root, step2_transform=step2_transform, overwrite=args.overwrite,
            )
            session_summaries.append(result["summary"])
        except Exception as exc:
            print(f"[{subject}/{session}] FAILED: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append({"subject": subject, "session": session, "error": str(exc)})

    out_root.mkdir(parents=True, exist_ok=True)

    if session_summaries:
        summary_columns = [
            "subject", "session",
            "literature_style_peak_mean_lr_contrast", "literature_style_peak_z",
            "left_peak_contrast", "left_peak_z",
            "right_peak_contrast", "right_peak_z",
        ]
        results_df = pd.DataFrame(session_summaries)[summary_columns]
        out_csv = out_root / f"{subject}_all_sessions_summary.csv"
        results_df.to_csv(out_csv, index=False)
        print(f"\nSaved summary: {out_csv}")
        print(results_df.to_string(index=False))
    else:
        print("\nNo sessions completed successfully.")

    if failures:
        print(f"\n{len(failures)} session(s) failed or were skipped:")
        for f in failures:
            print(f"  {f['subject']}/{f['session']}: {f['error']}")
        failures_csv = out_root / f"{subject}_failed_sessions.csv"
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
        print(f"Saved failure log: {failures_csv}")


if __name__ == "__main__":
    main()
