# Done by ChatGPT

#!/usr/bin/env bash
set -uo pipefail
#command -v dcm2niix
# Convert all subjects found under RAW_ROOT from DICOM to NIfTI.
#
# Expected input structure:
# RAW_ROOT/
# ├── sub002/
# │   ├── adaptation/
# │   └── experimental/
# ├── sub003/
# └── ...
#
# Run:
#   bash convert_all_subjects.sh
#
# Optional:
#   chmod +x convert_all_subjects.sh
#   ./convert_all_subjects.sh

RAW_ROOT="/home/maria/Documents/data/hiwi_sample/tmp-maria/raw_sample_dcm"
OUT_ROOT="/home/maria/Documents/data/hiwi_sample/tmp-maria/raw_sample_nifti"

# Check that dcm2niix is installed and available.
if ! command -v dcm2niix >/dev/null 2>&1; then
    echo "ERROR: dcm2niix was not found in PATH."
    echo "Activate the environment containing dcm2niix or install it first."
    exit 1
fi

# Check that the source directory exists.
if [ ! -d "$RAW_ROOT" ]; then
    echo "ERROR: RAW_ROOT does not exist:"
    echo "$RAW_ROOT"
    exit 1
fi

mkdir -p "$OUT_ROOT"

# Find every subject directory directly under RAW_ROOT.
mapfile -t SUBJECT_DIRS < <(
    find "$RAW_ROOT" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name "sub*" \
        | sort
)

if [ "${#SUBJECT_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no subject folders matching 'sub*' were found in:"
    echo "$RAW_ROOT"
    exit 1
fi

TOTAL_SUBJECTS="${#SUBJECT_DIRS[@]}"
SUCCESSFUL_SUBJECTS=0
FAILED_SUBJECTS=0
FAILED_LIST=()

convert_session() {
    local SUB="$1"
    local RAW="$2"
    local OUT="$3"
    local PHASE="$4"
    local SES="$5"

    local PHASE_DIR="$RAW/$PHASE"

    echo ""
    echo "====================================="
    echo "Subject: $SUB"
    echo "Phase:   $PHASE"
    echo "Session: $SES"
    echo "====================================="

    if [ ! -d "$PHASE_DIR" ]; then
        echo "WARNING: phase folder not found:"
        echo "$PHASE_DIR"
        return 0
    fi

    # Find all folders containing files whose path belongs to this session.
    mapfile -t DICOM_DIRS < <(
        find "$PHASE_DIR" \
            -type f \
            -path "*$SES*" \
            -printf "%h\n" \
            | sort -u
    )

    if [ "${#DICOM_DIRS[@]}" -eq 0 ]; then
        echo "WARNING: no DICOM folders found for $SES in:"
        echo "$PHASE_DIR"
        return 0
    fi

    echo "Found ${#DICOM_DIRS[@]} DICOM folder(s) for $SES"

    local SESSION_FAILED=0

    for DICOM_DIR in "${DICOM_DIRS[@]}"; do
        # Preserve the DICOM directory structure relative to the subject.
        local REL_DIR="${DICOM_DIR#"$RAW"/}"
        local OUT_DIR="$OUT/$REL_DIR"

        mkdir -p "$OUT_DIR"

        echo ""
        echo "Converting:"
        echo "Input:  $DICOM_DIR"
        echo "Output: $OUT_DIR"

        if dcm2niix \
            -z y \
            -f "${SUB}_${SES}_%p_%s" \
            -o "$OUT_DIR" \
            "$DICOM_DIR"
        then
            echo "SUCCESS: $DICOM_DIR"
        else
            echo "ERROR: conversion failed:"
            echo "$DICOM_DIR"
            SESSION_FAILED=1
        fi
    done

    return "$SESSION_FAILED"
}

convert_subject() {
    local SUBJECT_DIR="$1"
    local SUB
    SUB="$(basename "$SUBJECT_DIR")"

    local RAW="$SUBJECT_DIR"
    local OUT="$OUT_ROOT/$SUB"

    mkdir -p "$OUT"

    echo ""
    echo ""
    echo "#####################################"
    echo "SUBJECT: $SUB"
    echo "#####################################"

    local SUBJECT_FAILED=0

    echo ""
    echo "===== ADAPTATION ====="

    for SES in ses001 ses002; do
        if ! convert_session \
            "$SUB" \
            "$RAW" \
            "$OUT" \
            "adaptation" \
            "$SES"
        then
            SUBJECT_FAILED=1
        fi
    done

    echo ""
    echo "===== EXPERIMENTAL ====="

    for SES in ses003 ses004 ses005 ses006; do
        if ! convert_session \
            "$SUB" \
            "$RAW" \
            "$OUT" \
            "experimental" \
            "$SES"
        then
            SUBJECT_FAILED=1
        fi
    done

    echo ""
    if [ "$SUBJECT_FAILED" -eq 0 ]; then
        echo "SUBJECT FINISHED: $SUB"
        echo "Output: $OUT"
        return 0
    fi

    echo "SUBJECT FINISHED WITH ERRORS: $SUB"
    echo "Output: $OUT"
    return 1
}

echo "====================================="
echo "DICOM TO NIFTI CONVERSION"
echo "====================================="
echo "Raw root:    $RAW_ROOT"
echo "Output root: $OUT_ROOT"
echo "Subjects:    $TOTAL_SUBJECTS"
echo "dcm2niix:    $(command -v dcm2niix)"
echo "====================================="

for SUBJECT_DIR in "${SUBJECT_DIRS[@]}"; do
    SUB="$(basename "$SUBJECT_DIR")"

    if convert_subject "$SUBJECT_DIR"; then
        SUCCESSFUL_SUBJECTS=$((SUCCESSFUL_SUBJECTS + 1))
    else
        FAILED_SUBJECTS=$((FAILED_SUBJECTS + 1))
        FAILED_LIST+=("$SUB")
    fi
done

echo ""
echo ""
echo "====================================="
echo "FINAL SUMMARY"
echo "====================================="
echo "Total subjects:      $TOTAL_SUBJECTS"
echo "Completed subjects:  $SUCCESSFUL_SUBJECTS"
echo "Subjects with errors: $FAILED_SUBJECTS"
echo "Output saved under:"
echo "$OUT_ROOT"

if [ "$FAILED_SUBJECTS" -gt 0 ]; then
    echo ""
    echo "Subjects with conversion errors:"
    printf '  %s\n' "${FAILED_LIST[@]}"
    exit 1
fi

echo ""
echo "ALL SUBJECTS FINISHED SUCCESSFULLY"