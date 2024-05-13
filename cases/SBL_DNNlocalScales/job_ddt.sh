#!/bin/bash
#PBS -N dbug
### Project code
#PBS -A UCLB0017 
#PBS -l walltime=02:00:00
#PBS -q develop
### Merge output and error files
#PBS -j oe
#PBS -k eod
#PBS -l select=1:ncpus=64:mpiprocs=64
### Specify index range of sub-jobs
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M ac5006@columbia.edu

source ~/env_cpptorch
conda activate cpptorch

sh clean.sh

ddt --connect ./microhh run SBL1800 
