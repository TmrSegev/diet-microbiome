#!/bin/bash
#$ -N train_models          # Job name
#$ -cwd                        # Run from current working dir
#$ -o logs/diet_0.out          # Stdout path
#$ -e logs/diet_0.err          # Stderr path
#$ -pe threads 8
#$ -l h_rt=48:00:00            # Max runtime
#$ -l h_vmem=32G                # Memory per core

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export N_THREADS=8

echo START
/usr/wisdom/python3/bin/python /net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/code/train_models.py
echo FINISH