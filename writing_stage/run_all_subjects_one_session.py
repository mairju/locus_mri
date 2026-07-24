#!/usr/bin/env python3

"""
python3 -m lc_pipeline.run_all_subjects_one_session --subjects all -session ses001

"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from ..lc_pipeline import config
from ..lc_pipeline.data_discovery import build_session_dataframe
from ..lc_pipeline.run_subject import run_session


def discover_subjects(root_nifti: Path) -> list[str]:

    if not root_nifti.exists():
        raise FileNotFoundError(
            f"Root NIfTI directory does not exist: {root_nifti}"
        )

    subjects = sorted(
        path.name
        for path in root_nifti.glob("sub*")
        if path.is_dir()
    )

    if not subjects:
        raise RuntimeError(
            f"No subject directories matching 'sub*' were found under:\n"
            f"{root_nifti}"
        )

    return subjects


def path_is_missing(value) -> bool:

    if value is None or value is False:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    return False


def save_progress(
    summaries: list[dict],
    failures: list[dict],
    *,
    session: str,
    out_root: Path,
) -> tuple[Path | None, Path | None]:
    """
    Save the current progress after each subject.

    This is useful because the registrations can take a long time. If the
    script stops later, results from previously completed subjects remain
    available.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = None
    failure_csv = None

    if summaries:
        results_df = pd.DataFrame(summaries)

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

        # Keep the important columns first, but do not fail if Step 8 changes.
        first_columns = [
            column
            for column in preferred_columns
            if column in results_df.columns
        ]

        remaining_columns = [
            column
            for column in results_df.columns
            if column not in first_columns
        ]

        results_df = results_df[first_columns + remaining_columns]

        summary_csv = out_root / f"{session}_all_subjects_summary.csv"
        results_df.to_csv(summary_csv, index=False)

    if failures:
        failure_csv = out_root / f"{session}_failed_subjects.csv"
        pd.DataFrame(failures).to_csv(failure_csv, index=False)

    return summary_csv, failure_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            "Subjects to process, for example: sub002 sub003. "
            "Use 'all' to discover every sub* folder."
        ),
    )

    parser.add_argument(
        "--session",
        default="ses001",
        help="Single session to process. Default: ses001",
    )

    parser.add_argument(
        "--root-nifti",
        default=str(config.ROOT_NIFTI),
        help="Root directory containing the subjects' NIfTI data.",
    )

    parser.add_argument(
        "--out-root",
        default=str(config.OUT_ROOT),
        help="Output/derivatives root directory.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute steps even when their outputs already exist.",
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
    session = args.session

    if "all" in args.subjects:
        if len(args.subjects) > 1:
            parser.error(
                "Use either '--subjects all' or provide explicit subjects, "
                "not both."
            )

        subjects = discover_subjects(root_nifti)
    else:
        subjects = sorted(set(args.subjects))

    step2_transform = (
        config.STEP2_TRANSFORM_FULL
        if args.full_syn
        else config.STEP2_TRANSFORM_QUICK
    )

    print("\nRun configuration")
    print("=" * 70)
    print(f"Session:          {session}")
    print(f"Number subjects: {len(subjects)}")
    print(f"Root NIfTI:       {root_nifti}")
    print(f"Output root:      {out_root}")
    print(f"Step 2 transform: {step2_transform}")
    print(f"Overwrite:        {args.overwrite}")
    print("=" * 70)

    print("\nSubjects:")
    for subject in subjects:
        print(f"  - {subject}")

    subject_summaries: list[dict] = []
    failures: list[dict] = []

    for subject_index, subject in enumerate(subjects, start=1):
        print(
            f"\n{'#' * 70}\n"
            f"Subject {subject_index}/{len(subjects)}: "
            f"{subject} / {session}\n"
            f"{'#' * 70}"
        )

        try:
            # Discover only the requested subject and session.
            df = build_session_dataframe(
                session=[session],
                subject=subject,
                root=root_nifti,
                output_root=out_root,
                save=True,
            )

            row_matches = df[
                (df["subject"] == subject)
                & (df["session"] == session)
            ]

            if row_matches.empty:
                error_message = "No matching dataframe row was discovered."

                print(
                    f"[{subject}/{session}] {error_message} Skipping.",
                    file=sys.stderr,
                )

                failures.append(
                    {
                        "subject": subject,
                        "session": session,
                        "error": error_message,
                    }
                )

                save_progress(
                    subject_summaries,
                    failures,
                    session=session,
                    out_root=out_root,
                )
                continue

            if len(row_matches) > 1:
                print(
                    f"[{subject}/{session}] WARNING: "
                    f"{len(row_matches)} rows were found. "
                    f"The first row will be used."
                )

            row = row_matches.iloc[0]

            t1w_path = row.get("t1w_path")
            tse_path = row.get("tse_path")

            if path_is_missing(t1w_path) or path_is_missing(tse_path):
                error_message = (
                    f"Missing T1w or TSE path. "
                    f"t1w_path={t1w_path!r}, tse_path={tse_path!r}"
                )

                print(
                    f"[{subject}/{session}] {error_message} Skipping.",
                    file=sys.stderr,
                )

                failures.append(
                    {
                        "subject": subject,
                        "session": session,
                        "error": error_message,
                    }
                )

                save_progress(
                    subject_summaries,
                    failures,
                    session=session,
                    out_root=out_root,
                )
                continue

            print(f"T1w: {t1w_path}")
            print(f"TSE: {tse_path}")

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

            # Make sure summary is stored as a normal dictionary.
            if isinstance(summary, pd.Series):
                summary = summary.to_dict()
            elif not isinstance(summary, dict):
                summary = dict(summary)

            # Protect against subject/session not being included by Step 8.
            summary["subject"] = subject
            summary["session"] = session

            subject_summaries.append(summary)

            print(f"\n[{subject}/{session}] COMPLETED")

        except Exception as exc:
            print(
                f"\n[{subject}/{session}] FAILED: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()

            failures.append(
                {
                    "subject": subject,
                    "session": session,
                    "error": str(exc),
                }
            )

        # Save after every subject so previous results are not lost.
        summary_csv, failure_csv = save_progress(
            subject_summaries,
            failures,
            session=session,
            out_root=out_root,
        )

        if summary_csv is not None:
            print(f"Current combined summary: {summary_csv}")

        if failure_csv is not None:
            print(f"Current failure log: {failure_csv}")

    print("\n" + "=" * 70)
    print("RUN FINISHED")
    print("=" * 70)
    print(f"Requested subjects: {len(subjects)}")
    print(f"Completed subjects: {len(subject_summaries)}")
    print(f"Failed/skipped:     {len(failures)}")

    summary_csv, failure_csv = save_progress(
        subject_summaries,
        failures,
        session=session,
        out_root=out_root,
    )

    if subject_summaries:
        results_df = pd.DataFrame(subject_summaries)

        display_columns = [
            column
            for column in [
                "subject",
                "session",
                "literature_style_peak_mean_lr_contrast",
                "literature_style_peak_z",
                "left_peak_contrast",
                "left_peak_z",
                "right_peak_contrast",
                "right_peak_z",
            ]
            if column in results_df.columns
        ]

        print("\nCompleted results:")
        print(results_df[display_columns].to_string(index=False))

        if summary_csv is not None:
            print(f"\nSaved combined summary:\n{summary_csv}")
    else:
        print("\nNo subjects completed successfully.")

    if failures:
        print("\nFailed or skipped subjects:")

        for failure in failures:
            print(
                f"  {failure['subject']}/{failure['session']}: "
                f"{failure['error']}"
            )

        if failure_csv is not None:
            print(f"\nSaved failure log:\n{failure_csv}")


if __name__ == "__main__":
    main()