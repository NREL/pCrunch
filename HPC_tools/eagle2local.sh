# ------ Transfer DLC Data from eagle runs -------


# --- 5MW LAND LEGACY ---
outdir='/projects/ssc/nabbas/DLC_Analysis/5MW_Land/5MW_Land_legacy/'
indir='../BatchOutputs/5MW_Land/5MW_Land_legacy/'
mkdir -p $indir;
scp nabbas@eagle.hpc.nrel.gov:$outdir*.outb $indir
scp nabbas@eagle.hpc.nrel.gov:$outdir/case_matrix.yaml $indir

# --- 5MW LAND ROSCO ---
outdir2='/projects/ssc/nabbas/DLC_Analysis/5MW_Land/5MW_Land_rosco/'
indir2='../BatchOutputs/5MW_Land/5MW_Land_ROSCO/'
mkdir -p $indir2;
scp nabbas@eagle.hpc.nrel.gov:$outdir2*.outb $indir2
scp nabbas@eagle.hpc.nrel.gov:$outdir2/case_matrix.yaml $indir2

# # --- 5MW OC3Spar LEGACY ---
# outdir3='/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar/5MW_OC3Spar_legacy/'
# indir3='../BatchOutputs/5MW_OC3Spar/5MW_OC3Spar_legacy/'
# mkdir -p $indir3;
# scp nabbas@eagle.hpc.nrel.gov:$outdir3*.outb $indir3
# scp nabbas@eagle.hpc.nrel.gov:$outdir3/case_matrix.yaml $indir3

# # --- 5MW OC3Spar ROSCO ---
# outdir4='/projects/ssc/nabbas/DLC_Analysis/5MW_OC3Spar/5MW_OC3Spar_ROSCO/'
# indir4='../BatchOutputs/5MW_OC3Spar/5MW_OC3Spar_ROSCO_2/'
# mkdir -p $indir4;
# scp nabbas@eagle.hpc.nrel.gov:$outdir4*.outb $indir4
# scp nabbas@eagle.hpc.nrel.gov:$outdir4/case_matrix.yaml $indir4
