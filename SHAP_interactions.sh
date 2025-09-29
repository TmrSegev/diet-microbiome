#!/bin/bash
#$ -N SHAP_interaction          # Job name
#$ -cwd                        # Run from current working dir
#$ -o logs/diet_0.out          # Stdout path
#$ -e logs/diet_0.err          # Stderr path
#$ -l h_rt=01:00:00            # Max runtime
#$ -l h_vmem=32G                # Memory per core

echo START
/usr/wisdom/python3/bin/python /net/mraid20/export/genie/LabData/Analyses/tomerse/diet_mb/code/SHAP_interactions.py
ECHO FINISH