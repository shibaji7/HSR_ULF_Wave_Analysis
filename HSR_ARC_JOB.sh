#!/bin/bash
#PBS -l nodes=1:ppn=48
#PBS -l walltime=24:00:00
#PBS -l procs=40
#PBS -q normal_q
#PBS -A solarflare
#PBS -W group_list=cascades
#PBS -M shibaji7@vt.edu
#PBS -m bea
cd $PBS_O_WORKDIR
module purge
module load Anaconda/5.2.0
source activate ulf
python py/pre_process.py
exit;