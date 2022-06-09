#!/bin/bash
#SBATCH -J HSR-ARC-CODE
#SBATCH --account=personal
#SBATCH --partition=normal_q
#SBATCH --ntasks=24
#SBATCH --time=0-01:00:00
#SBATCH --mem=10G

cd $SLURM_SUBMIT_DIR
module purge
module load Anaconda3
source activate ulf
python --version
python py/pre_process.py