#!/usr/bin/env python3
"""
python3 -m lc_pipeline.run_subject_all_sessions \
    --sessions ses001 \
    --out-root /home/maria/Documents/data/hiwi_sample/24.07_ses001 \
    --full-syn

python3 -m lc_pipeline.run_subject_all_sessions \
    --subject sub002 \
    --sessions ses001 \
    --out-root /home/maria/Documents/data/hiwi_sample/24.07_ses001 \
    --full-syn

python3 -m lc_pipeline.run_subject_all_sessions \
    --out-root /home/maria/Documents/data/hiwi_sample/24.07_ses001 \
    --full-syn
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

def discover_subjects(root_nifti: Path) -> list[str]:

    if not root_nifti.exists():
        raise FileNotFoundError(
            f"Root NIfTI directory does not exist:\n{root_nifti}"
        )

    if not root_nifti.is_dir():
        raise NotADirectoryError(
            f"Root NIfTI path is not a directory:\n{root_nifti}"
        )

    subjects = sorted(
        path.name
        for path in root_nifti.iterdir()
        if path.is_dir() and path.name.startswith("sub")
    )

    if not subjects:
        raise RuntimeError(
            "No subject folders starting with 'sub' were found in:\n"
            f"{root_nifti}"
        )

    return subjects


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--subject",
        "--subjects",
        dest="subjects",
        nargs="+",
        default=None,
        help=(
            "One or more subjects, e.g. --subject sub001 sub002. "
            "If omitted, all subject folders in root-nifti are used."
        ),
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help=(
            "One or more sessions, e.g. --sessions ses001 ses002. "
            "If omitted, all sessions in config.SESSIONS are used."
        ),
    )

    parser.add_argument(
        "--root-nifti",
        default=str(config.ROOT_NIFTI),
        help="Root directory containing the raw NIfTI subject folders.",
    )

    parser.add_argument(
        "--out-root",
        default=str(config.OUT_ROOT),
        help="Output/derivatives root directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute steps even if output files already exist.",
    )

    parser.add_argument(
        "--full-syn",
        action="store_true",
        help=(
            "Use antsRegistrationSyN[s] for Step 2 instead of "
            "the faster Quick preset."
        ),
    )

    args = parser.parse_args()

    root_nifti = Path(args.root_nifti)
    out_root = Path(args.out_root)

    subjects = (
        args.subjects
        if args.subjects
        else discover_subjects(root_nifti)
    )

    sessions = (
        args.sessions
        if args.sessions
        else list(config.SESSIONS.keys())
    )

    step2_transform = (
        config.STEP2_TRANSFORM_FULL
        if args.full_syn
        else config.STEP2_TRANSFORM_QUICK
    )

    print("=" * 70)
    print("LC pipeline")
    print("=" * 70)
    print(f"Root NIfTI:       {root_nifti}")
    print(f"Output root:      {out_root}")
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Sessions ({len(sessions)}): {sessions}")
    print(f"Step 2 transform: {step2_transform}")
    print(f"Overwrite:        {args.overwrite}")
    print("=" * 70)

    out_root.mkdir(parents=True, exist_ok=True)

    all_session_summaries = []
    all_failures = []

    for subject_index, subject in enumerate(subjects, start=1):
        print("\n" + "=" * 70)
        print(
            f"Subject {subject_index}/{len(subjects)}: {subject}"
        )
        print("=" * 70)

        try:
            df = build_session_dataframe(
                session=sessions,
                subject=subject,
                root=root_nifti,
                output_root=out_root,
                save=True,
            )

        except Exception as exc:
            print(
                f"[{subject}] Could not build session dataframe: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

            for session in sessions:
                all_failures.append(
                    {
                        "subject": subject,
                        "session": session,
                        "error": (
                            "could not build session dataframe: "
                            f"{exc}"
                        ),
                    }
                )

            # Continue with the next subject.
            continue

        subject_summaries = []
        subject_failures = []

        for session_index, session in enumerate(
            sessions,
            start=1,
        ):
            print(
                f"\n[{subject}/{session}] "
                f"Session {session_index}/{len(sessions)}"
            )

            row_matches = df[
                (df["subject"] == subject)
                & (df["session"] == session)
            ]

            if row_matches.empty:
                error_message = (
                    "no matching row in discovered dataframe"
                )

                print(
                    f"[{subject}/{session}] "
                    f"{error_message} -- skipping"
                )

                failure = {
                    "subject": subject,
                    "session": session,
                    "error": error_message,
                }

                subject_failures.append(failure)
                all_failures.append(failure)
                continue

            row = row_matches.iloc[0]

            t1w_path = row.get("t1w_path")
            tse_path = row.get("tse_path")

            if (
                not t1w_path
                or t1w_path is False
                or not tse_path
                or tse_path is False
            ):
                error_message = "missing t1w_path or tse_path"

                print(
                    f"[{subject}/{session}] "
                    f"{error_message} -- skipping"
                )

                failure = {
                    "subject": subject,
                    "session": session,
                    "error": error_message,
                }

                subject_failures.append(failure)
                all_failures.append(failure)
                continue

            try:
                result = run_session(
                    subject=subject,
                    session=session,
                    t1w_path=t1w_path,
                    tse_path=tse_path,
                    out_root=out_root,
                    step2_transform=step2_transform,
                    overwrite=args.overwrite,
                )

                summary = result["summary"]

                subject_summaries.append(summary)
                all_session_summaries.append(summary)

                print(
                    f"[{subject}/{session}] completed successfully"
                )

            except Exception as exc:
                print(
                    f"[{subject}/{session}] FAILED: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()

                failure = {
                    "subject": subject,
                    "session": session,
                    "error": str(exc),
                }

                subject_failures.append(failure)
                all_failures.append(failure)

        if subject_summaries:
            subject_results_df = pd.DataFrame(subject_summaries)

            preferred_columns = [
                "subject",
                "session",
                "literature_style_peak_mean_lr_contrast",
                "literature_style_peak_z",
                "left_peak_contrast",
                "left_peak_z",
                "right_peak_contrast",
                "right_peak_z",
            ]

            available_columns = [
                column
                for column in preferred_columns
                if column in subject_results_df.columns
            ]

            remaining_columns = [
                column
                for column in subject_results_df.columns
                if column not in available_columns
            ]

            subject_results_df = subject_results_df[
                available_columns + remaining_columns
            ]

            subject_summary_csv = (
                out_root
                / f"{subject}_all_sessions_summary.csv"
            )

            subject_results_df.to_csv(
                subject_summary_csv,
                index=False,
            )

            print(
                f"\nSaved subject summary: "
                f"{subject_summary_csv}"
            )
            print(subject_results_df.to_string(index=False))

        else:
            print(
                f"\nNo sessions completed successfully "
                f"for {subject}."
            )

        # Save one failure log per subject.
        if subject_failures:
            subject_failures_csv = (
                out_root
                / f"{subject}_failed_sessions.csv"
            )

            pd.DataFrame(subject_failures).to_csv(
                subject_failures_csv,
                index=False,
            )

            print(
                f"\n{len(subject_failures)} session(s) failed "
                f"or were skipped for {subject}."
            )
            print(
                f"Saved subject failure log: "
                f"{subject_failures_csv}"
            )

    if all_session_summaries:
        all_results_df = pd.DataFrame(all_session_summaries)

        preferred_columns = [
            "subject",
            "session",
            "literature_style_peak_mean_lr_contrast",
            "literature_style_peak_z",
            "left_peak_contrast",
            "left_peak_z",
            "right_peak_contrast",
            "right_peak_z",
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in all_results_df.columns
        ]

        remaining_columns = [
            column
            for column in all_results_df.columns
            if column not in available_columns
        ]

        all_results_df = all_results_df[
            available_columns + remaining_columns
        ]

        combined_summary_csv = (
            out_root
            / "all_subjects_all_sessions_summary.csv"
        )

        all_results_df.to_csv(
            combined_summary_csv,
            index=False,
        )

        print("\n" + "=" * 70)
        print("Combined results")
        print("=" * 70)
        print(f"Saved combined summary: {combined_summary_csv}")
        print(all_results_df.to_string(index=False))

    else:
        print("\nNo subject/session completed successfully.")

    if all_failures:
        combined_failures_csv = (
            out_root
            / "all_subjects_failed_sessions.csv"
        )

        pd.DataFrame(all_failures).to_csv(
            combined_failures_csv,
            index=False,
        )

        print(
            f"\n{len(all_failures)} total session(s) failed "
            "or were skipped."
        )
        print(
            f"Saved combined failure log: "
            f"{combined_failures_csv}"
        )

    successful_count = len(all_session_summaries)
    failed_count = len(all_failures)

    print("\n" + "=" * 70)
    print("Pipeline finished")
    print("=" * 70)
    print(f"Successful sessions: {successful_count}")
    print(f"Failed/skipped:      {failed_count}")


if __name__ == "__main__":
    main()