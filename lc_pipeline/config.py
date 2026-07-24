"""
Pipeline configuration.

python -m lc_pipeline.run_subject_all_sessions \
    --subject sub002

python -m lc_pipeline.run_subject_all_sessions \
    --subject sub002

python -m lc_pipeline.run_subject_all_sessions \
    --subject sub017 \
    --sessions ses001 \
    --full-syn

echo $FSLDIR
"""
from pathlib import Path

# ---- directores FIXED ----
#EXP_ID = "exp02_testing_code_step05" 
ROOT_NIFTI = Path("/home/maria/Documents/data/hiwi_sample/tmp-maria/raw_sample_nifti")
OUT_ROOT = Path(f"/home/maria/Documents/data/hiwi_sample")
#OUT_ROOT = Path("/media/maria/A91A-44D3/mri_hiwi/debbuging/exp_{EXP_ID}")
FSL_DIR = Path("/home/maria/fsl")


ATLASES_ROOT = Path("/home/maria/Documents/projects/mri_studies/atlases/")

MNI_TEMPLATE = ATLASES_ROOT / "MNI152_T1_0.5mm.nii.gz"

# Dahl et al. 2022 LC meta-masks
LC_MASK_LEFT = ATLASES_ROOT / "LCmetaMask_left_MNI05_s01f_plus50.nii.gz"
LC_MASK_RIGHT = ATLASES_ROOT / "LCmetaMask_right_MNI05_s01f_plus50.nii.gz"
LC_MASK_BOTH = ATLASES_ROOT / "LCmetaMask_MNI05_s01f_plus50.nii.gz"

# Dahl et al. dorsal pontine tegmentum reference mask
DPT_MASK = ATLASES_ROOT / "CentralReferenceMask_MNI05.nii.gz"

BRAINSTEM_MASK_MNI_PATH = ATLASES_ROOT / "brainstem_mask_HarvardOxford_MNI_1mm.nii.gz"

SESSIONS = {
    "ses001": "adaptation",
    "ses002": "adaptation",
    "ses003": "experimental",
    "ses004": "experimental",
    "ses005": "experimental",
    "ses006": "experimental",
}

DEFAULT_KEYWORDS = ["T1w", "TSE"]

TARGET_SPACING = (0.5, 0.5, 0.5) 
CONVERGENCE = {"iters": [50, 50, 30, 20], "tol": 1e-6}
SHRINK_FACTOR = 2
SPLINE_PARAMETER = 200


STEP2_TRANSFORM_FULL = "antsRegistrationSyN[s]"
STEP2_TRANSFORM_QUICK = "antsRegistrationSyNQuick[s]"
STEP2_TRANSFORM = STEP2_TRANSFORM_QUICK  

BRAINSTEM_DILATION_RADIUS_VOX = 3  
BRAINSTEM_DILATION_ITERS_FSL = 3   
CROP_PADDING_VOX = 10              
