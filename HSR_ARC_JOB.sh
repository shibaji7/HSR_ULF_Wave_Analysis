#!/bin/bash
#SBATCH -J HSR-ARC-CODE
#SBATCH --account=personal
#SBATCH --partition=v100_normal_q
#SBATCH --nodes=3 --ntasks-per-node=12 --cpus-per-task=1
#SBATCH --time=0-12:10:00
#SBATCH --mem=20G

module purge
module load Anaconda/5.2.0
source activate ulf
cd $SLURM_SUBMIT_DIR