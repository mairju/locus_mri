#!/usr/bin/env python3
"""
- Writes copy_report.csv.

report ( copy_report.csv.):
    COPIED
    DRY_RUN
    MISSING
    NOT_COPIED

python copy_selected_dcm_folders.py \
    --subjects all \
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable

SRC_ROOT = Path(
    "/mnt/neurostorage/neuroStorage2/SLOWED/data/MRI/raw"
)

DST_ROOT = Path(
    "/media/guest/mairj/all_subjects_selected_mri_raw"
)


SESSIONS = {
    "ses001": "adaptation",
    "ses002": "adaptation",
    "ses003": "experimental",
    "ses004": "experimental",
    "ses005": "experimental",
    "ses006": "experimental",
}

DEFAULT_KEYWORDS = [
    "T1w",
    "TSE",
]

REPORT_COLUMNS = [
    "subject",
    "session",
    "phase",
    "keyword",
    "status",
    "source",
    "destination",
]

VALID_STATUSES = {
    "COPIED",
    "DRY_RUN",
    "MISSING",
    "NOT_COPIED",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected T1w/TSE DICOM folders while preserving "
            "the source directory structure."
        )
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help=(
            'Subjects to process, for example "sub002 sub003", '
            'or "all". Default: all.'
        ),
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help=(
            "Sequence-folder keywords to copy. "
            "Default: T1w TSE."
        ),
    )

    parser.add_argument(
        "--src-root",
        type=Path,
        default=SRC_ROOT,
        help=f"Source root. Default: {SRC_ROOT}",
    )

    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DST_ROOT,
        help=f"Destination root. Default: {DST_ROOT}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying data.",
    )

    return parser.parse_args()


def normalize_subject(subject: str) -> str:
    """
    Accept forms such as:
        sub002
        sub-002
        002
        2

    Return the most likely dataset form:
        sub002
    """
    value = subject.strip()

    if value.lower() == "all":
        return "all"

    lowered = value.lower()

    if lowered.startswith("sub-"):
        suffix = value[4:]
    elif lowered.startswith("sub"):
        suffix = value[3:]
    else:
        suffix = value

    if suffix.isdigit():
        return f"sub{int(suffix):03d}"

    return value


def discover_subjects(src_root: Path) -> list[str]:
    """
    Find subject directories directly below SRC_ROOT.
    """
    subjects = sorted(
        path.name
        for path in src_root.iterdir()
        if path.is_dir()
        and path.name.lower().startswith("sub")
    )

    return subjects


def resolve_subjects(
    requested_subjects: Iterable[str],
    src_root: Path,
) -> list[str]:
    normalized = [
        normalize_subject(subject)
        for subject in requested_subjects
    ]

    if "all" in normalized:
        return discover_subjects(src_root)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(normalized))


def find_session_root(
    phase_path: Path,
    session: str,
) -> Path | None:
    """
    Find the acquisition root whose directory name starts with
    'neuropsychology_slowed' and whose full path belongs to the
    requested session.

    Expected structure resembles:

        subject/
          adaptation or experimental/
            slowed_sub002_ses001_.../
              neuropsychology_slowed_1_.../
                T1w/
                TSE/
    """
    if not phase_path.exists():
        return None

    candidates = sorted(
        path
        for path in phase_path.rglob("*")
        if path.is_dir()
        and path.name.lower().startswith(
            "neuropsychology_slowed"
        )
        and session.lower() in str(path).lower()
    )

    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            f"  Warning: found {len(candidates)} possible roots "
            f"for {session}; using:\n"
            f"    {candidates[0]}"
        )

    return candidates[0]


def find_matching_folders(
    session_root: Path,
    keyword: str,
) -> list[Path]:
    """
    Match only immediate child folders of the acquisition root.

    Matching is case-insensitive and checks whether the keyword
    appears in the folder name.
    """
    keyword_lower = keyword.lower()

    return sorted(
        child
        for child in session_root.iterdir()
        if child.is_dir()
        and keyword_lower in child.name.lower()
    )


def add_report_row(
    rows: list[dict[str, str]],
    *,
    subject: str,
    session: str,
    phase: str,
    keyword: str,
    status: str,
    source: Path | str,
    destination: Path | str,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid report status: {status}"
        )

    rows.append(
        {
            "subject": subject,
            "session": session,
            "phase": phase,
            "keyword": keyword,
            "status": status,
            "source": str(source),
            "destination": str(destination),
        }
    )


def copy_sequence_folder(
    source_folder: Path,
    *,
    src_root: Path,
    dst_root: Path,
    dry_run: bool,
) -> tuple[str, Path]:
    """
    Copy one sequence folder while preserving its path relative
    to SRC_ROOT.
    """
    relative_path = source_folder.relative_to(src_root)
    destination_folder = dst_root / relative_path

    if destination_folder.exists():
        return "NOT_COPIED", destination_folder

    if dry_run:
        return "DRY_RUN", destination_folder

    destination_folder.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        source_folder,
        destination_folder,
        copy_function=shutil.copy2,
    )

    return "COPIED", destination_folder


def write_report(
    rows: list[dict[str, str]],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=REPORT_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: list[dict[str, str]],
    report_path: Path,
) -> None:
    counts = {
        status: sum(
            row["status"] == status
            for row in rows
        )
        for status in sorted(VALID_STATUSES)
    }

    print("\n" + "=" * 60)
    print("COPY SUMMARY")
    print("=" * 60)

    for status, count in counts.items():
        print(f"{status:12s}: {count}")

    print(f"\nReport: {report_path}")


def main() -> None:
    args = parse_arguments()

    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(
            f"Source root does not exist:\n{src_root}"
        )

    if not src_root.is_dir():
        raise NotADirectoryError(
            f"Source root is not a directory:\n{src_root}"
        )

    subjects = resolve_subjects(
        args.subjects,
        src_root,
    )

    keywords = list(
        dict.fromkeys(args.keywords)
    )

    if not subjects:
        raise RuntimeError(
            f"No subject folders were found under:\n{src_root}"
        )

    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Subjects:    {len(subjects)}")
    print(f"Keywords:    {', '.join(keywords)}")
    print(f"Dry run:     {args.dry_run}")

    report_rows: list[dict[str, str]] = []

    for subject in subjects:
        subject_path = src_root / subject

        print("\n" + "-" * 60)
        print(f"Subject: {subject}")
        print("-" * 60)

        if not subject_path.exists():
            print(f"  Missing subject directory: {subject_path}")

            for session, phase in SESSIONS.items():
                for keyword in keywords:
                    add_report_row(
                        report_rows,
                        subject=subject,
                        session=session,
                        phase=phase,
                        keyword=keyword,
                        status="MISSING",
                        source=subject_path,
                        destination="",
                    )

            continue

        for session, phase in SESSIONS.items():
            phase_path = subject_path / phase

            print(
                f"\n  {session} [{phase}]"
            )

            session_root = find_session_root(
                phase_path,
                session,
            )

            if session_root is None:
                print(
                    "    Session acquisition root not found."
                )

                for keyword in keywords:
                    add_report_row(
                        report_rows,
                        subject=subject,
                        session=session,
                        phase=phase,
                        keyword=keyword,
                        status="MISSING",
                        source=phase_path,
                        destination="",
                    )

                continue

            print(
                f"    Acquisition root: {session_root}"
            )

            for keyword in keywords:
                matching_folders = find_matching_folders(
                    session_root,
                    keyword,
                )

                if not matching_folders:
                    print(
                        f"    {keyword}: MISSING"
                    )

                    add_report_row(
                        report_rows,
                        subject=subject,
                        session=session,
                        phase=phase,
                        keyword=keyword,
                        status="MISSING",
                        source=session_root,
                        destination="",
                    )

                    continue

                for source_folder in matching_folders:
                    status, destination_folder = (
                        copy_sequence_folder(
                            source_folder,
                            src_root=src_root,
                            dst_root=dst_root,
                            dry_run=args.dry_run,
                        )
                    )

                    print(
                        f"    {keyword}: {status}\n"
                        f"      from: {source_folder}\n"
                        f"      to:   {destination_folder}"
                    )

                    add_report_row(
                        report_rows,
                        subject=subject,
                        session=session,
                        phase=phase,
                        keyword=keyword,
                        status=status,
                        source=source_folder,
                        destination=destination_folder,
                    )

    report_path = (
        dst_root / "copy_report.csv"
    )

    write_report(
        report_rows,
        report_path,
    )

    print_summary(
        report_rows,
        report_path,
    )


if __name__ == "__main__":
    main()