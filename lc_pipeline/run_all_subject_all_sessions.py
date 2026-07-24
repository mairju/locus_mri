#!/usr/bin/env python3
"""
python3 -m lc_pipeline.run_all_subjects_all_sessions

python3 -m lc_pipeline.run_all_subjects_all_sessions --subjects sub001 sub002 sub003

python3 -m lc_pipeline.run_all_subjects_all_sessions --subjects sub001 sub002 --sessions ses001 ses002


python3 -m lc_pipeline.run_all_subjects_all_sessions --overwrite

python3 -m lc_pipeline.run_all_subjects_all_sessions --subjects sub001 sub017 --sessions ses001 ses002 --full-syn



### Updates:

python3 -m lc_pipeline.run_all_subjects_all_sessions --subjects sub001 sub017 --sessions ses001 ses002 --id exp_01 --full-syn


## to run all subs and all sessions
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses001 --id exp_24.07_ses001 --full-syn
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses002 --id exp_24.07_ses002 --full-syn
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses003 --id exp_24.07_ses003 --full-syn
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses004 --id exp_24.07_ses004 --full-syn
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses005 --id exp_24.07_ses005 --full-syn
python3 -m lc_pipeline.run_all_subjects_all_sessions --sessions ses006 --id exp_24.07_ses006 --full-syn
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import config


def discover_subjects(root_nifti: Path) -> list[str]:

    if not root_nifti.exists():
        raise FileNotFoundError(
            f"Root NIfTI directory does not exist:\n{root_nifti}"
        )

    subjects = sorted(
        path.name
        for path in root_nifti.iterdir()
        if path.is_dir() and path.name.startswith("sub")
    )

    if not subjects:
        raise RuntimeError(
            f"No subject folders starting with 'sub' were found in:\n"
            f"{root_nifti}"
        )

    return subjects

def run_all_subjects(
    subjects: list[str],
    sessions: list[str] | None = None,
    root_nifti: Path | None = None,
    out_root: Path | None = None,
    overwrite: bool = False,
    full_syn: bool = False,
    stop_on_error: bool = False,
) -> dict:

    successful_subjects: list[str] = []
    failed_subjects: dict[str, int] = {}

    total_subjects = len(subjects)

    for subject_number, subject in enumerate(subjects, start=1):

        print("\n")
        print("=" * 80)
        print(
            f"SUBJECT {subject_number}/{total_subjects}: {subject}"
        )
        print("=" * 80)

        command = [
            sys.executable,
            "-m",
            "lc_pipeline.run_subject_all_sessions",
            "--subject",
            subject,
        ]

        if sessions:
            command.append("--sessions")
            command.extend(sessions)

        if root_nifti is not None:
            command.extend(
                ["--root-nifti", str(root_nifti)]
            )

        if out_root is not None:
            command.extend(
                ["--out-root", str(out_root)]
            )

        if overwrite:
            command.append("--overwrite")

        if full_syn:
            command.append("--full-syn")

        print("Running command:")
        print(" ".join(command))
        print()

        completed_process = subprocess.run(
            command,
            check=False,
        )

        if completed_process.returncode == 0:
            successful_subjects.append(subject)
            print(f"\nCompleted successfully: {subject}")

        else:
            failed_subjects[subject] = completed_process.returncode

            print(
                f"\nFAILED: {subject} "
                f"(exit code {completed_process.returncode})"
            )

            if stop_on_error:
                print("Stopping because --stop-on-error was used.")
                break

    return {
        "successful_subjects": successful_subjects,
        "failed_subjects": failed_subjects,
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--id",
        required=True,
        help=(
            "Experiment ID used as the output folder name. "
            "Example: exp_01"
        ),
    )

    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=(
            "Subjects to process, for example: sub001 sub002. "
            "Default: automatically discover all subject folders."
        ),
    )

    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help=(
            "Sessions to process, for example: ses001 ses002. "
            "Default: use all sessions configured in config.py."
        ),
    )

    parser.add_argument(
        "--root-nifti",
        default=str(config.ROOT_NIFTI),
        help="Root directory containing the subject folders.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs that already exist.",
    )

    parser.add_argument(
        "--full-syn",
        action="store_true",
        help="Use full SyN instead of the quicker registration.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one subject fails.",
    )

    args = parser.parse_args()

    root_nifti = Path(args.root_nifti)

    experiment_id = args.id.strip()

    if not experiment_id:
        parser.error("--id cannot be empty")

    if "/" in experiment_id or "\\" in experiment_id:
        parser.error(
            "--id must be a folder name, not a path. "
            "Example: --id exp_01"
        )

    out_root = config.OUT_ROOT / experiment_id
    out_root.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(root_nifti)

    print("\nExperiment configuration:")
    print(f"  Experiment ID: {experiment_id}")
    print(f"  Input root:    {root_nifti}")
    print(f"  Output root:   {out_root}")

    print("\nSubjects that will be processed:")
    for subject in subjects:
        print(f"  - {subject}")

    if args.sessions:
        print("\nSessions that will be processed:")
        for session in args.sessions:
            print(f"  - {session}")
    else:
        print("\nSessions: all sessions from config.SESSIONS")

    result = run_all_subjects(
        subjects=subjects,
        sessions=args.sessions,
        root_nifti=root_nifti,
        out_root=out_root,
        overwrite=args.overwrite,
        full_syn=args.full_syn,
        stop_on_error=args.stop_on_error,
    )

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {out_root}")

    print(
        f"\nSuccessful subjects: "
        f"{len(result['successful_subjects'])}"
    )

    for subject in result["successful_subjects"]:
        print(f"  ✓ {subject}")

    print(
        f"\nFailed subjects: "
        f"{len(result['failed_subjects'])}"
    )

    for subject, exit_code in result["failed_subjects"].items():
        print(f"  ✗ {subject}: exit code {exit_code}")

    if result["failed_subjects"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()