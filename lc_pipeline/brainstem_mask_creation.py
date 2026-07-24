#from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


"""
Annotations for RtDs:

It creates a binary brainstem mask from the standard atlas 1mm (from FSL, HarvardOxford). The resolution of the brainstem mask will change after 
registering it to the MNI space. The brain-stem mask in the Harvard-Oxford subcortical atlas is index 7.

Refereces:
https://git.fmrib.ox.ac.uk/fsl/data_atlases/-/blob/FinalFive/HarvardOxford-Subcortical.xml

"""

def find_and_configure_fsl() -> Path:
    result = subprocess.run(
        ["find", os.environ.get("HOME", "/root"), "/usr/local", "/opt",
         "-path", "*/etc/fslconf/fsl.sh"],
        capture_output=True, text=True,
    )
    config_files = [Path(p.strip()) for p in result.stdout.splitlines() if p.strip()]

    if not config_files:
        raise RuntimeError("Could not find an FSL installation.")

    # searching for the correct index
    candidates = []
    for config_file in config_files:
        fsl_root = config_file.parents[2]
        atlas_path = fsl_root / "data" / "atlases" / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
        atlas_xml = fsl_root / "data" / "atlases" / "HarvardOxford-Subcortical.xml"
        if atlas_path.exists() and atlas_xml.exists():
            candidates.append(fsl_root)

    if not candidates:
        raise RuntimeError(
            "FSL configuration files were found, but no installation contained the required Harvard-Oxford atlas."
        )

    candidates.sort(key=lambda path: ("/pkgs/" in str(path) or "/src/" in str(path), len(path.parts)))
    fsl_dir = candidates[0]

    os.environ["FSLDIR"] = str(fsl_dir)
    os.environ.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    for executable_dir in [fsl_dir / "share" / "fsl" / "bin", fsl_dir / "bin"]:
        if executable_dir.exists():
            os.environ["PATH"] = str(executable_dir) + os.pathsep + os.environ.get("PATH", "")

    print("Detected FSLDIR:", fsl_dir)
    print("fslmaths:", shutil.which("fslmaths"))
    print("fslstats:", shutil.which("fslstats"))
    return fsl_dir


def create_brainstem_mask(output_path, overwrite: bool = False) -> Path:
    fsl_dir = find_and_configure_fsl()
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        return output_path

    fsldir_value = fsl_dir or os.environ.get("FSLDIR")
    if not fsldir_value:
        raise EnvironmentError("FSLDIR is not defined. Load FSL before running this.")
    fsldir = Path(fsldir_value)

    atlas_path = fsldir / "data" / "atlases" / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
    atlas_xml = fsldir / "data" / "atlases" / "HarvardOxford-Subcortical.xml"

    if not atlas_path.exists():
        raise FileNotFoundError(f"Harvard-Oxford atlas not found: {atlas_path}")
    if not atlas_xml.exists():
        raise FileNotFoundError(f"Harvard-Oxford XML not found: {atlas_xml}")

    tree = ET.parse(atlas_xml)
    matching_labels = [
        label for label in tree.findall(".//label")
        if (label.text or "").strip().lower() in {"brain-stem", "brainstem", "brain stem"}
    ]

    if len(matching_labels) != 1:
        found_names = [(label.text or "").strip() for label in tree.findall(".//label")]
        raise RuntimeError(
            f"Could not uniquely identify the Brain-Stem label. Matching labels: {matching_labels}\n"
            f"Available labels: {found_names}"
        )

    atlas_index = int(matching_labels[0].attrib["index"])
    label_value = atlas_index + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["fslmaths", str(atlas_path), "-thr", str(label_value), "-uthr", str(label_value),
         "-bin", str(output_path)],
        check=True,
    )

    stats = subprocess.run(["fslstats", str(output_path), "-V"], check=True, capture_output=True, text=True)
    voxel_count = int(float(stats.stdout.strip().split()[0]))

    if voxel_count == 0:
        raise RuntimeError(f"The generated brainstem mask is empty. Selected atlas value: {label_value}")

    print(f"Brain-Stem atlas index: {atlas_index}")
    print(f"Max-probability image value: {label_value}")
    print(f"Brainstem mask voxels: {voxel_count}")
    print(f"Saved: {output_path}")
    return output_path
