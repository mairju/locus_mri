#!/usr/bin/env python3
"""
python3 -m lc_pipeline.run_all_subjects_all_sessions \
    --id exp_01 \
    --subjects sub001 sub017 \
    --sessions ses001 ses002 \
    --jobs 2 \
    --threads-per-job 4 \
    --full-syn
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import datetime
from pathlib import Path

from . import config


def discover_subjects(root_nifti: Path) -> list[str]:
    """
    Discover subject directories inside the raw NIfTI root.

    Expected subject directory names:
        sub001
        sub002
        ...
    """

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
            "No subject folders starting with 'sub' were found in:\n"
            f"{root_nifti}"
        )

    return subjects


def build_subject_command(
    *,
    subject: str,
    sessions: list[str] | None,
    root_nifti: Path,
    out_root: Path,
    overwrite: bool,
    full_syn: bool,
) -> list[str]:
    """Build the subprocess command for one subject."""

    command = [
        sys.executable,
        "-m",
        "lc_pipeline.run_subject_all_sessions",
        "--subject",
        subject,
        "--root-nifti",
        str(root_nifti),
        "--out-root",
        str(out_root),
    ]

    if sessions:
        command.append("--sessions")
        command.extend(sessions)

    if overwrite:
        command.append("--overwrite")

    if full_syn:
        command.append("--full-syn")

    return command


def create_worker_environment(
    threads_per_job: int,
) -> dict[str, str]:
    """
    Create the environment used by each subject subprocess.

    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS controls threading in ANTs/ITK.

    BLAS thread counts are limited to one because the main parallelism is:
        subjects × ANTs threads

    This helps avoid nested thread oversubscription.
    """

    environment = os.environ.copy()

    environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(
        threads_per_job
    )

    environment["OMP_NUM_THREADS"] = str(threads_per_job)

    # Avoid NumPy/BLAS starting additional nested thread pools.
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"

    return environment


def run_one_subject(
    *,
    subject: str,
    sessions: list[str] | None,
    root_nifti: Path,
    out_root: Path,
    overwrite: bool,
    full_syn: bool,
    threads_per_job: int,
    subject_number: int,
    total_subjects: int,
) -> dict:
    """
    Run the pipeline for one subject.

    Output from each subject is written to an independent log file so that
    messages from parallel subprocesses do not become mixed together.
    """

    command = build_subject_command(
        subject=subject,
        sessions=sessions,
        root_nifti=root_nifti,
        out_root=out_root,
        overwrite=overwrite,
        full_syn=full_syn,
    )

    environment = create_worker_environment(
        threads_per_job=threads_per_job
    )

    log_directory = out_root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    log_path = log_directory / f"{subject}.log"

    start_time = time.perf_counter()
    start_datetime = datetime.now()

    print(
        f"\n▶ Starting {subject} "
        f"({subject_number}/{total_subjects})",
        flush=True,
    )
    print(
        f"  Threads: {threads_per_job}",
        flush=True,
    )
    print(
        f"  Log:     {log_path}",
        flush=True,
    )

    try:
        with log_path.open(
            "a",
            encoding="utf-8",
        ) as log_file:

            log_file.write("\n")
            log_file.write("=" * 80)
            log_file.write("\n")

            log_file.write(
                f"Started: {start_datetime.isoformat()}\n"
            )

            log_file.write(
                f"Subject: {subject}\n"
            )

            log_file.write(
                f"Threads: {threads_per_job}\n"
            )

            log_file.write(
                f"Command: {shlex.join(command)}\n"
            )

            log_file.write("=" * 80)
            log_file.write("\n\n")
            log_file.flush()

            completed_process = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                env=environment,
            )

            return_code = completed_process.returncode

            elapsed_seconds = (
                time.perf_counter() - start_time
            )

            log_file.write("\n")
            log_file.write("=" * 80)
            log_file.write("\n")

            log_file.write(
                f"Finished: {datetime.now().isoformat()}\n"
            )

            log_file.write(
                f"Return code: {return_code}\n"
            )

            log_file.write(
                f"Elapsed seconds: {elapsed_seconds:.1f}\n"
            )

            log_file.write("=" * 80)
            log_file.write("\n")

    except Exception as error:
        elapsed_seconds = time.perf_counter() - start_time

        return {
            "subject": subject,
            "return_code": -1,
            "elapsed_seconds": elapsed_seconds,
            "log_path": log_path,
            "error": str(error),
        }

    return {
        "subject": subject,
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "log_path": log_path,
        "error": None,
    }


def run_all_subjects(
    *,
    subjects: list[str],
    sessions: list[str] | None,
    root_nifti: Path,
    out_root: Path,
    overwrite: bool = False,
    full_syn: bool = False,
    stop_on_error: bool = False,
    jobs: int = 2,
    threads_per_job: int = 4,
) -> dict:
    """
    Run several subjects concurrently.

    Parameters
    ----------
    jobs
        Maximum number of subject subprocesses running simultaneously.

    threads_per_job
        Maximum number of ITK/ANTs threads assigned to each subject process.
    """

    if not subjects:
        raise ValueError("The subject list is empty.")

    if jobs < 1:
        raise ValueError("jobs must be at least 1.")

    if threads_per_job < 1:
        raise ValueError(
            "threads_per_job must be at least 1."
        )

    total_subjects = len(subjects)

    # There is no reason to create more workers than subjects.
    jobs = min(jobs, total_subjects)

    successful_subjects: list[str] = []
    failed_subjects: dict[str, int] = {}
    subject_results: dict[str, dict] = {}

    print("\nParallel-processing configuration:")
    print(f"  Parallel subject jobs: {jobs}")
    print(f"  Threads per job:       {threads_per_job}")
    print(
        f"  Maximum assigned threads: "
        f"{jobs * threads_per_job}"
    )
    print(
        f"  Logical CPUs detected: "
        f"{os.cpu_count() or 'unknown'}"
    )

    subject_iterator = iter(
        enumerate(subjects, start=1)
    )

    running_futures: dict[
        Future,
        tuple[int, str],
    ] = {}

    stop_scheduling = False

    with ThreadPoolExecutor(
        max_workers=jobs,
        thread_name_prefix="lc-subject",
    ) as executor:

        def submit_next_subject() -> bool:
            """
            Submit the next subject.

            Returns False when there are no subjects left.
            """

            try:
                subject_number, subject = next(
                    subject_iterator
                )
            except StopIteration:
                return False

            future = executor.submit(
                run_one_subject,
                subject=subject,
                sessions=sessions,
                root_nifti=root_nifti,
                out_root=out_root,
                overwrite=overwrite,
                full_syn=full_syn,
                threads_per_job=threads_per_job,
                subject_number=subject_number,
                total_subjects=total_subjects,
            )

            running_futures[future] = (
                subject_number,
                subject,
            )

            return True

        # Initially fill all worker slots.
        for _ in range(jobs):
            if not submit_next_subject():
                break

        while running_futures:
            completed_futures, _ = wait(
                running_futures,
                return_when=FIRST_COMPLETED,
            )

            for future in completed_futures:
                _, subject = running_futures.pop(
                    future
                )

                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "subject": subject,
                        "return_code": -1,
                        "elapsed_seconds": 0.0,
                        "log_path": (
                            out_root
                            / "logs"
                            / f"{subject}.log"
                        ),
                        "error": str(error),
                    }

                subject_results[subject] = result

                elapsed_minutes = (
                    result["elapsed_seconds"] / 60
                )

                if result["return_code"] == 0:
                    successful_subjects.append(subject)

                    print(
                        f"\n✓ Completed {subject} "
                        f"in {elapsed_minutes:.1f} minutes",
                        flush=True,
                    )

                else:
                    failed_subjects[subject] = (
                        result["return_code"]
                    )

                    print(
                        f"\n✗ FAILED {subject}",
                        flush=True,
                    )

                    print(
                        f"  Exit code: "
                        f"{result['return_code']}",
                        flush=True,
                    )

                    print(
                        f"  Log: {result['log_path']}",
                        flush=True,
                    )

                    if result["error"]:
                        print(
                            f"  Error: {result['error']}",
                            flush=True,
                        )

                    if stop_on_error:
                        stop_scheduling = True

                        print(
                            "\nNo additional subjects will "
                            "be started because "
                            "--stop-on-error was used.",
                            flush=True,
                        )

                # Fill the worker slot with the next subject,
                # unless stop_on_error has been triggered.
                if not stop_scheduling:
                    submit_next_subject()

    # Keep final results in the original subject order.
    successful_subjects = [
        subject
        for subject in subjects
        if subject in successful_subjects
    ]

    failed_subjects = {
        subject: failed_subjects[subject]
        for subject in subjects
        if subject in failed_subjects
    }

    skipped_subjects = [
        subject
        for subject in subjects
        if subject not in subject_results
    ]

    return {
        "successful_subjects": successful_subjects,
        "failed_subjects": failed_subjects,
        "skipped_subjects": skipped_subjects,
        "subject_results": subject_results,
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
            "Subjects to process, for example: "
            "sub001 sub002. "
            "Default: automatically discover all "
            "subject folders."
        ),
    )

    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help=(
            "Sessions to process, for example: "
            "ses001 ses002. "
            "Default: use all sessions from config.SESSIONS."
        ),
    )

    parser.add_argument(
        "--root-nifti",
        default=str(config.ROOT_NIFTI),
        help=(
            "Root directory containing the subject folders."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help=(
            "Number of subjects to process simultaneously. "
            "Default: 2."
        ),
    )

    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=4,
        help=(
            "Number of ANTs/ITK threads assigned to each "
            "subject process. Default: 4."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs that already exist.",
    )

    parser.add_argument(
        "--full-syn",
        action="store_true",
        help=(
            "Use full SyN instead of the quicker registration."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Do not start additional subjects after a failure. "
            "Subjects already running will finish."
        ),
    )

    args = parser.parse_args()

    root_nifti = Path(args.root_nifti)

    experiment_id = args.id.strip()

    if not experiment_id:
        parser.error("--id cannot be empty.")

    if "/" in experiment_id or "\\" in experiment_id:
        parser.error(
            "--id must be a folder name, not a path. "
            "Example: --id exp_01"
        )

    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")

    if args.threads_per_job < 1:
        parser.error(
            "--threads-per-job must be at least 1."
        )

    out_root = (
        config.OUT_ROOT
        / experiment_id
    )

    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(root_nifti)

    if args.sessions:
        invalid_sessions = [
            session
            for session in args.sessions
            if session not in config.SESSIONS
        ]

        if invalid_sessions:
            parser.error(
                "Unknown sessions: "
                + ", ".join(invalid_sessions)
                + ". Configured sessions are: "
                + ", ".join(config.SESSIONS)
            )

    print("\n")
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)

    print(f"Experiment ID:     {experiment_id}")
    print(f"Input root:        {root_nifti}")
    print(f"Output root:       {out_root}")
    print(f"Parallel jobs:     {args.jobs}")
    print(
        f"Threads per job:   "
        f"{args.threads_per_job}"
    )
    print(
        f"Registration:      "
        f"{'full SyN' if args.full_syn else 'quick SyN'}"
    )

    print("\nSubjects that will be processed:")

    for subject in subjects:
        print(f"  - {subject}")

    if args.sessions:
        print("\nSessions that will be processed:")

        for session in args.sessions:
            print(f"  - {session}")
    else:
        print(
            "\nSessions: all sessions from config.SESSIONS"
        )

    start_time = time.perf_counter()

    result = run_all_subjects(
        subjects=subjects,
        sessions=args.sessions,
        root_nifti=root_nifti,
        out_root=out_root,
        overwrite=args.overwrite,
        full_syn=args.full_syn,
        stop_on_error=args.stop_on_error,
        jobs=args.jobs,
        threads_per_job=args.threads_per_job,
    )

    total_elapsed_minutes = (
        time.perf_counter() - start_time
    ) / 60

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Experiment ID:    {experiment_id}")
    print(f"Output directory: {out_root}")
    print(
        f"Total runtime:    "
        f"{total_elapsed_minutes:.1f} minutes"
    )

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

    for subject, exit_code in (
        result["failed_subjects"].items()
    ):
        print(
            f"  ✗ {subject}: exit code {exit_code}"
        )

    print(
        f"\nSkipped subjects: "
        f"{len(result['skipped_subjects'])}"
    )

    for subject in result["skipped_subjects"]:
        print(f"  - {subject}")

    print(
        f"\nSubject logs: {out_root / 'logs'}"
    )

    if result["failed_subjects"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()